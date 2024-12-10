# Source/Networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class LinearNetwork(nn.Module):
    """
    A simple utility MLP block: Linear + optional activation + dropout.
    Used by state and action encoders.
    """
    def __init__(self, in_features: int, out_features: int, dropout_rate=0.0, activation=True):
        super(LinearNetwork, self).__init__()
        layers = [nn.Linear(in_features, out_features)]
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        if activation:
            layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*layers)

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return self.model(x)


class StatesEncoder(nn.Module):
    """
    Encodes per-agent observations into embeddings.
    Input: (Steps,Env,Agents,Obs_dim)
    Output: (Steps,Env,Agents,H)
    Applies a simple MLP (LinearNetwork) along the last dimension.
    """
    def __init__(self, input_size, output_size, dropout_rate=0.0, activation=True):
        super(StatesEncoder, self).__init__()
        self.fc = LinearNetwork(input_size, output_size, dropout_rate, activation)

    def forward(self, x):
        # fc is applied to the last dimension (Obs_dim -> H)
        return self.fc(x)


class StatesActionsEncoder(nn.Module):
    """
    Encodes groupmates' states and actions together into embeddings.
    Input: 
      - observation: (S,E,A,(A-1),H)
      - action: (S,E,A,(A-1),Action_dim)
    Output:
      (S,E,A,(A-1),H') where H' = output_size
    Concatenate obs+action along last dim, then LinearNetwork.
    """
    def __init__(self, state_dim, action_dim, output_size, dropout_rate=0.0, activation=True):
        super(StatesActionsEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim + action_dim, output_size, dropout_rate, activation)

    def forward(self, observation, action):
        # Last dim: H + Action_dim -> fc
        x = torch.cat([observation, action], dim=-1)
        return self.fc(x)


class ValueNetwork(nn.Module):
    """
    A small MLP for producing a scalar value (for V or baseline).
    Input: (Batch,Features)
    Output: (Batch,1)
    """
    def __init__(self, in_features: int, hidden_size: int, dropout_rate=0.0):
        super(ValueNetwork, self).__init__()
        self.value_net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.Dropout(p=dropout_rate),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, 1)
        )
        for layer in self.value_net:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight)

    def forward(self, x):
        return self.value_net(x)


class RSA(nn.Module):
    """
    Relational Self-Attention (RSA):
    Used to produce relational embeddings among agents.

    Input: (Batch,Agents,H)
    Output: (Batch,Agents,H) with attention-enhanced features.
    """
    def __init__(self, embed_size, heads, dropout_rate=0.0):
        super(RSA, self).__init__()
        self.query_embed = self.linear_layer(embed_size, embed_size)
        self.key_embed = self.linear_layer(embed_size, embed_size)
        self.value_embed = self.linear_layer(embed_size, embed_size)
        
        self.input_norm = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)
        
        self.multihead_attn = nn.MultiheadAttention(embed_size, heads, batch_first=True, dropout=dropout_rate)

    def linear_layer(self, input_size: int, output_size: int) -> nn.Module:
        layer = nn.Linear(input_size, output_size)
        init.kaiming_normal_(layer.weight)
        return layer

    def forward(self, x: torch.Tensor):
        # x: (Batch,Agents,H)
        x = self.input_norm(x)
        q = self.query_embed(x)
        k = self.key_embed(x)
        v = self.value_embed(x)
        output, _ = self.multihead_attn(q, k, v)
        output = x + output  # residual connection
        return self.output_norm(output)


class MultiAgentEmbeddingNetwork(nn.Module):
    """
    MultiAgentEmbeddingNetwork:
    - Encodes per-agent obs with StatesEncoder (g()).
    - Applies RSA to produce relational embeddings.
    - Provides methods to split agent from groupmates and encode groupmates obs+actions.

    Steps:
    1) obs -> agent_obs_encoder -> (S,E,A,H)
    2) RSA over agents -> (S,E,A,H)
    """
    def __init__(self, config):
        super(MultiAgentEmbeddingNetwork, self).__init__()
        net_cfg = config['agent']['params']['networks']['MultiAgentEmbeddingNetwork']

        self.agent_obs_encoder = StatesEncoder(**net_cfg['agent_obs_encoder'])
        self.agent_embedding_encoder = StatesEncoder(**net_cfg['agent_embedding_encoder'])
        self.obs_action_encoder = StatesActionsEncoder(**net_cfg['obs_actions_encoder'])

        rsa_cfg = net_cfg['RSA']
        self.rsa = RSA(**rsa_cfg)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs: torch.Tensor):
        # obs: (S,E,A,obs_dim)
        # Encode per-agent obs:
        emb = self.agent_obs_encoder(obs) # (S,E,A,H)
        S,E,A,H = emb.shape
        flat = emb.view(S*E,A,H)
        attended = self.rsa(flat) # (S*E,A,H)
        return attended.view(S,E,A,H)

    def split_agent_obs_groupmates_obs_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        # Similar logic as before:
        # obs: (S,E,A,obs_dim), actions: (S,E,A,action_dim)
        S,E,A,obs_dim = obs.shape
        _,_,_,act_dim = actions.shape
        agent_obs_list = []
        groupmates_obs_list = []
        groupmates_actions_list = []

        for i in range(A):
            agent_obs_list.append(obs[:, :, i, :].unsqueeze(2))
            g_obs = torch.cat([obs[:, :, :i, :], obs[:, :, i+1:, :]], dim=2).unsqueeze(2)
            g_acts = torch.cat([actions[:, :, :i, :], actions[:, :, i+1:, :]], dim=2).unsqueeze(2)
            groupmates_obs_list.append(g_obs)
            groupmates_actions_list.append(g_acts)

        agent_obs = torch.cat(agent_obs_list, dim=2).unsqueeze(3).contiguous() 
        # (S,E,A,1,obs_dim)
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        # (S,E,A,(A-1),obs_dim)
        groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        # (S,E,A,(A-1),action_dim)
        return agent_obs, groupmates_obs, groupmates_actions

    def split_agent_groupmates_obs(self, agent_embeddings: torch.Tensor):
        # agent_embeddings: (S,E,A,H)
        S,E,A,H = agent_embeddings.shape
        agent_emb_list = []
        groupmates_emb_list = []
        for i in range(A):
            agent_emb_list.append(agent_embeddings[:, :, i, :].unsqueeze(2))
            g_emb = torch.cat([agent_embeddings[:, :, :i, :], agent_embeddings[:, :, i+1:, :]], dim=2).unsqueeze(2)
            groupmates_emb_list.append(g_emb)

        agent_embeds = torch.cat(agent_emb_list, dim=2).unsqueeze(3).contiguous()
        # (S,E,A,1,H)
        groupmates_embeds = torch.cat(groupmates_emb_list, dim=2).contiguous()
        # (S,E,A,(A-1),H)
        return agent_embeds, groupmates_embeds

    def encode_groupmates_obs_actions(self, groupmates_embeddings: torch.Tensor, groupmates_actions: torch.Tensor):
        # groupmates_embeddings: (S,E,A,(A-1),H)
        # groupmates_actions: (S,E,A,(A-1),Action_dim)
        return self.obs_action_encoder(groupmates_embeddings, groupmates_actions)


class SharedCritic(nn.Module):
    """
    SharedCritic with two heads:
    - value_head: V(s)
    - baseline_head: per-agent baseline Q_ψ for counterfactual baseline

    value(s): average embeddings over agents -> (S,E,H) -> value_head -> (S,E,1)
    baselines(x): x=(S,E,A,NA,H), average over NA dimension -> (S,E,A,H) -> baseline_head -> (S,E,A,1)
    """
    def __init__(self, config):
        super(SharedCritic, self).__init__()
        net_cfg = config['agent']['params']['networks']['critic_network']
        self.value_head = ValueNetwork(**net_cfg['value_head'])
        self.baseline_head = ValueNetwork(**net_cfg['baseline_head'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def values(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S,E,A,H)
        S,E,A,H = x.shape
        mean_emb = x.mean(dim=2) # (S,E,H)
        flat = mean_emb.view(S*E,H)
        vals = self.value_head(flat).view(S,E,1)
        return vals

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        # x: (S,E,A,A,H)
        # average over the "groupmates" dimension: (S,E,A,H)
        S,E,A,A2,H = x.shape
        assert A == A2
        mean_x = x.mean(dim=3) # (S,E,A,H)
        flat = mean_x.view(S*E*A,H)
        baseline_vals = self.baseline_head(flat).view(S,E,A,1)
        return baseline_vals


class ContinuousPolicyNetwork(nn.Module):
    """
    Continuous policy: outputs mean and std for actions.
    Input: (Batch,H)
    Output: mean:(Batch,action_dim), std:(Batch,action_dim)
    """
    def __init__(self, in_features: int, out_features: int, hidden_size: int):
        super(ContinuousPolicyNetwork, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU()
        )
        self.mean = nn.Linear(hidden_size, out_features)
        self.log_std = nn.Parameter(torch.zeros(out_features))

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
        init.xavier_normal_(self.mean.weight)

    def forward(self, x):
        features = self.shared(x)
        mean = torch.tanh(self.mean(features))  # actions in (-1,1)
        log_std = self.log_std.unsqueeze(0).expand_as(mean)
        std = torch.exp(log_std)
        return mean, std


class QNetwork(nn.Module):
    """
    Optional Q-network for Q^\pi(s,a) if needed.
    Input: state_dim + action_dim
    Output: scalar Q-value
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)
