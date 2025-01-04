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

# -----------------------------------------------------------------------
# AgentIDPosEnc: learnable embeddings for agent-index using sinusoidal features
# -----------------------------------------------------------------------
class AgentIDPosEnc(nn.Module):
    """
    For each integer agent index i, produce a learned embedding vector of size id_embed_dim,
    via a sinusoidal feature extraction + linear projection.
    """

    def __init__(self, num_freqs: int = 8, id_embed_dim: int = 32):
        """
        :param num_freqs: how many sinusoidal frequencies to use
        :param id_embed_dim: dimension of the final ID embedding
        """
        super().__init__()
        self.num_freqs = num_freqs
        self.id_embed_dim = id_embed_dim

        # The linear layer that maps 2*num_freqs features -> id_embed_dim
        self.linear = nn.Linear(2 * self.num_freqs, self.id_embed_dim)
        init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def sinusoidal_features(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        agent_ids: (NS, NE, NA), integer agent indices
        returns: (NS, NE, NA, 2 * num_freqs) float features from sin/cos expansions
        """
        # Convert agent IDs to float
        # shape => (NS, NE, NA)
        agent_ids_f = agent_ids.to(dtype=torch.float32)

        # We'll define some frequency scales, e.g. 2^0..2^(num_freqs-1)
        freq_exponents = torch.arange(self.num_freqs, device=agent_ids.device, dtype=torch.float32)
        scales = torch.pow(2.0, freq_exponents)  # shape => (num_freqs,)

        # Expand agent_ids_f and scales to broadcast for sin/cos
        # agent_ids_f => (NS, NE, NA) => unsqueeze => (NS, NE, NA, 1)
        agent_ids_f = agent_ids_f.unsqueeze(-1)  # shape => (NS, NE, NA, 1)
        # scales => (num_freqs,) => (1, num_freqs) for broadcast
        scales = scales.view(1, self.num_freqs)  # shape => (1, num_freqs)

        # We want a final shape => (NS, NE, NA, num_freqs)
        # We'll do agent_ids_f * scales => broadcast along last dimension
        # First, flatten out the (NS,NE,NA)->(X) dimension if needed
        # but we can also do a simpler method: repeat_interleave or expand:
        # Let's do a quick trick with outer product style broadcasting:

        # We'll produce sin/cos in a single step:
        multiplied = agent_ids_f * scales  # shape => (NS, NE, NA, num_freqs)
        sines = torch.sin(multiplied)
        cosines = torch.cos(multiplied)

        # Concatenate along last dimension => shape => (NS, NE, NA, 2*num_freqs)
        feats = torch.cat([sines, cosines], dim=-1)
        return feats

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        agent_ids: (NS, NE, NA) integer indices
        returns: (NS, NE, NA, id_embed_dim)
        """
        # 1) Compute raw sinusoidal features => (NS, NE, NA, 2*num_freqs)
        raw_feat = self.sinusoidal_features(agent_ids)  # shape => (..., 2*num_freqs)

        # 2) Flatten to pass through linear
        # shape => (NS,NE,NA, 2*num_freqs) => (NS*NE*NA, 2*num_freqs)
        NS, NE, NA, F = raw_feat.shape
        flattened = raw_feat.view(NS * NE * NA, F)

        # 3) linear => shape => (NS*NE*NA, id_embed_dim)
        embed = self.linear(flattened)

        # 4) reshape => (NS, NE, NA, id_embed_dim)
        embed = embed.view(NS, NE, NA, self.id_embed_dim)
        return embed

# -----------------------------------------------------------------------

class StatesEncoder(nn.Module):
    """
    Encodes per-agent observations into embeddings.
    Input: (Steps,Env,Agents,Obs_dim)
    Output: (Steps,Env,Agents,H)
    """
    def __init__(self, input_size, output_size, dropout_rate=0.0, activation=True):
        super(StatesEncoder, self).__init__()
        self.fc = LinearNetwork(input_size, output_size, dropout_rate, activation)

    def forward(self, x):
        return self.fc(x)

class StatesActionsEncoder(nn.Module):
    """
    Encodes groupmates' states and actions together into embeddings.
    Input: 
      - observation: (S,E,A,(A-1),H)
      - action: (S,E,A,(A-1),Action_dim)
    Output:
      (S,E,A,(A-1),H') where H' = output_size
    """
    def __init__(self, state_dim, action_dim, output_size, dropout_rate=0.0, activation=True):
        super(StatesActionsEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim + action_dim, output_size, dropout_rate, activation)

    def forward(self, observation, action):
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

# -----------------------------------------------------------------------
# MultiAgentEmbeddingNetwork with agent-ID positional enc
# -----------------------------------------------------------------------
class MultiAgentEmbeddingNetwork(nn.Module):
    """
    MultiAgentEmbeddingNetwork:
    - Encodes per-agent obs with StatesEncoder (g()).
    - Adds Agent ID positional enc, fusing ID embedding with obs embedding
    - Applies RSA to produce relational embeddings.
    - Provides methods to isolate agent-groupmates obs for baseline networks.
    """
    def __init__(self, net_cfg):
        super(MultiAgentEmbeddingNetwork, self).__init__()

        # Normal obs enc
        self.agent_obs_encoder = StatesEncoder(**net_cfg['agent_obs_encoder'])

        # Extra aggregator if needed
        self.agent_embedding_encoder = StatesEncoder(**net_cfg['agent_embedding_encoder'])

        # For groupmates obs+action encoding
        self.obs_action_encoder = StatesActionsEncoder(**net_cfg['obs_actions_encoder'])

        # RSA for attending to multi-agents
        self.rsa = RSA(**net_cfg['RSA'])

        # Id_encoder for agent IDs
        self.id_encoder = AgentIDPosEnc(**net_cfg['agent_id_enc'])

        H = net_cfg['agent_obs_encoder']['output_size']
        id_emb_dim = net_cfg['agent_id_enc']['id_embed_dim']

        self.fuse = nn.Linear(H + id_emb_dim, H)
        init.kaiming_normal_(self.fuse.weight)
        nn.init.constant_(self.fuse.bias, 0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, obs: torch.Tensor):
        """
        obs: (S,E,A,obs_dim)
        Steps:
          1) Build agent_ids => (S,E,A)
          2) ID embed => (S,E,A,id_embed_dim)
          3) obs embed => (S,E,A,H)
          4) cat => fuse => RSA => final
        """
        S, E, A, obs_dim = obs.shape
        obs_device = obs.device

        # 1) Build agent_ids
        agent_ids = torch.arange(A, device=obs_device).view(1,1,A)
        # shape => (S,E,A), broadcast
        agent_ids = agent_ids.expand(S, E, A)

        # 2) ID embed => (S,E,A,id_embed_dim)
        id_emb = self.id_encoder(agent_ids)

        # 3) obs embed => (S,E,A,H)
        obs_emb = self.agent_obs_encoder(obs)

        # 4) fuse
        combined = torch.cat([obs_emb, id_emb], dim=-1)  # => (S,E,A, H+id_embed_dim)
        fused = self.fuse(combined)                      # => (S,E,A,H)

        # 5) apply RSA
        flat = fused.view(S*E, A, -1)                    # (S*E, A, H)
        attended = self.rsa(flat)                        # => (S*E, A, H)
        return attended.view(S, E, A, -1)

    def split_agent_obs_groupmates_obs_actions(self, obs: torch.Tensor, actions: torch.Tensor):
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
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        return agent_obs, groupmates_obs, groupmates_actions

    def split_agent_groupmates_obs(self, agent_embeddings: torch.Tensor):
        S,E,A,H = agent_embeddings.shape
        agent_emb_list = []
        groupmates_emb_list = []
        for i in range(A):
            agent_emb_list.append(agent_embeddings[:, :, i, :].unsqueeze(2))
            g_emb = torch.cat([agent_embeddings[:, :, :i, :], agent_embeddings[:, :, i+1:, :]], dim=2).unsqueeze(2)
            groupmates_emb_list.append(g_emb)

        agent_embeds = torch.cat(agent_emb_list, dim=2).unsqueeze(3).contiguous()
        groupmates_embeds = torch.cat(groupmates_emb_list, dim=2).contiguous()
        return agent_embeds, groupmates_embeds

    def encode_groupmates_obs_actions(self, groupmates_embeddings: torch.Tensor, groupmates_actions: torch.Tensor):
        return self.obs_action_encoder(groupmates_embeddings, groupmates_actions)

class SharedCritic(nn.Module):
    """
    SharedCritic with two heads:
    - value_head: V(s)
    - baseline_head: per-agent baseline Q_ψ for counterfactual baseline
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
        S,E,A,A2,H = x.shape
        assert A == A2
        mean_x = x.mean(dim=3) # (S,E,A,H)
        flat = mean_x.view(S*E*A,H)
        baseline_vals = self.baseline_head(flat).view(S,E,A,1)
        return baseline_vals
    
class DiscretePolicyNetwork(nn.Module):
    """
    A policy network for discrete action spaces.
    If out_features is an int => single-categorical
    If out_features is a list => multi-discrete
    """
    def __init__(self, in_features: int, out_features, hidden_size: int):
        super().__init__()
        self.in_features = in_features
        self.hidden_size = hidden_size

        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )

        # differentiate single vs multi
        if isinstance(out_features, int):
            # single categorical => one head
            self.head = nn.Linear(hidden_size, out_features)
        else:
            # multi-discrete => separate heads for each branch
            self.branch_heads = nn.ModuleList([
                nn.Linear(hidden_size, branch_size)
                for branch_size in out_features
            ])
            self.out_is_multi = True

        # init weights
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        if hasattr(self, 'head'):
            init.xavier_normal_(self.head.weight)
            nn.init.constant_(self.head.bias, 0.0)
        else:
            for lin in self.branch_heads:
                init.xavier_normal_(lin.weight)
                nn.init.constant_(lin.bias, 0.0)

    def forward(self, x: torch.Tensor):
        """
        x => (B, in_features)
        returns => either (B, out_features) if single cat
                or list of (B, branch_size) if multi-discrete
        """
        feats = self.shared(x)
        if hasattr(self, 'head'):
            # single cat
            return self.head(feats)   # shape => (B, out_features)
        else:
            # multi-discrete
            return [branch(feats) for branch in self.branch_heads]

class ContinuousPolicyNetwork(nn.Module):
    """
    Continuous policy network that outputs raw (unbounded) mean and clamped log_std.
    """
    def __init__(self, in_features: int, out_features: int, hidden_size: int, 
                 log_std_min: float, log_std_max: float):
        super(ContinuousPolicyNetwork, self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU()
        )
        self.mean = nn.Linear(hidden_size, out_features)
        self.log_std_layer = nn.Linear(hidden_size, out_features)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                init.xavier_normal_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

        init.xavier_normal_(self.mean.weight)
        nn.init.constant_(self.mean.bias, 0.0)

        init.xavier_normal_(self.log_std_layer.weight)
        nn.init.constant_(self.log_std_layer.bias, 0.0)

    def forward(self, x):
        features = self.shared(x)
        mean = self.mean(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
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
