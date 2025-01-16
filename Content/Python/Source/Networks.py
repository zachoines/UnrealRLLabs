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

class StatesEncoder(nn.Module):
    """
    Encodes per-agent observations into embeddings.
    Input: (Steps,Env,Agents,Obs_dim)
    Output: (Steps,Env,Agents,H)
    """
    def __init__(self, state_dim, output_size, dropout_rate=0.0, activation=True):
        super(StatesEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim, output_size, dropout_rate, activation)

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

class MultiAgentEmbeddingNetwork(nn.Module):
    """
    MultiAgentEmbeddingNetwork:
      - g() => Encodes per-agent obs (Eqn 6 or the policy eqn).
      - f() => Encodes (obs_j, action_j) for groupmates in baseline eqn (Eqn 8).
      - RSA => a relational self-attention block for multi-agent embeddings.
      - AgentIDPosEnc => optional agent ID embeddings for uniqueness.

    We provide distinct methods for:
      1) embed_obs_for_value(...)   -> feed into V( RSA(g(o^i)) )
      2) embed_obs_for_policy(...)  -> feed into pi( RSA(g(o^i)) )
      3) embed_obs_actions_for_baseline(...) -> feed into Q( RSA([g(o^i), f(o^j,a^j)]_{j != i}) )

    Important note on agent ID usage:
      - We append ID embeddings *before* we do splits or state encodings,
        so each agent's ID remains properly aligned with its partial obs.
    """

    def __init__(self, net_cfg):
        super(MultiAgentEmbeddingNetwork, self).__init__()

        # 1) read user configs
        obs_enc_cfg   = net_cfg["agent_obs_encoder"]
        acts_enc_cfg  = net_cfg["obs_actions_encoder"]
        id_enc_cfg    = net_cfg["agent_id_enc"]
        rsa_cfg       = net_cfg["RSA"]
        pre_obs_cfg   = net_cfg["pre_obs_encoder"]
        id_emb_dim    = id_enc_cfg["id_embed_dim"]

        # 2) auto-correct input size for g():
        old_g_in = obs_enc_cfg["state_dim"]
        new_g_in = old_g_in + id_emb_dim
        obs_enc_cfg["state_dim"] = new_g_in

        # 3) auto-correct input size for f():
        old_f_in = acts_enc_cfg["state_dim"]
        new_f_in = old_f_in + id_emb_dim
        acts_enc_cfg["state_dim"] = new_f_in

        # 4) Now create the submodules with updated sizes
        self.g = StatesEncoder(**obs_enc_cfg)
        self.f = StatesActionsEncoder(**acts_enc_cfg)
        self.rsa = RSA(**rsa_cfg)
        self.id_encoder = AgentIDPosEnc(**id_enc_cfg)
        self.pre_obs_transform = LinearNetwork(**pre_obs_cfg)

    def embed_obs_for_policy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        eqn(Policy):  π( RSA(g(o^i)) )
        1) Run pre_obs_transform on obs
        2) Add agent IDs into the obs (before g).
        3) pass obs->g->RSA
        => shape (S,E,A,H)
        """
        return self._obs_with_id_and_rsa(obs)

    def embed_obs_for_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        eqn(6):  V( RSA(g(o^i)) )
        Identical pipeline if the environment doesn't differentiate 
        between how policy and value embed states.
        => shape (S,E,A,H)
        """
        return self._obs_with_id_and_rsa(obs)

    def embed_obs_actions_for_baseline(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        eqn(8):  Q_ψ( RSA( [g(o^i), f(o^j,a^j)]_{ j != i }) )
        We want shape => (S,E,A,A,H).

        Steps:
         1) pre_obs_transform + agent IDs
         2) split => agent_obs_i, groupmates_obs_j, groupmates_actions_j
         3) g(agent_obs_i), f(groupmates_obs_j, groupmates_actions_j)
         4) cat => (S,E,A,A,H)
         5) RSA => (S,E,A,A,H)
        """
        # 1) Pre-transform + ID
        obs_with_id = self._add_id_emb(obs)

        # 2) split => agent_obs, groupmates_obs, groupmates_actions
        agent_obs, groupmates_obs, groupmates_actions = \
            self.split_agent_obs_groupmates_obs_actions(obs_with_id, actions)

        # 3) g(...) => agent_obs => (S,E,A,1,H)
        agent_obs_emb = self.g(agent_obs)

        # 4) f(...) => groupmates_obs_actions => (S,E,A,(A-1),H)
        groupmates_emb = self.f(groupmates_obs, groupmates_actions)

        # => cat => (S,E,A,A,H)
        combined = torch.cat([agent_obs_emb, groupmates_emb], dim=3)

        # Flatten (S,E,A) => batch dimension, and A => the agent dimension
        S, E, A, A2, H = combined.shape
        combined_reshaped = combined.view(S*E*A, A2, H)

        attended = self.rsa(combined_reshaped)
        attended = attended.view(S, E, A, A2, H)
        return attended

    def _add_id_emb(self, obs: torch.Tensor) -> torch.Tensor:
        """
        1) pre_obs_transform on obs
        2) agent_ids => (S,E,A)
        3) pass through self.id_encoder => (S,E,A,id_emb_dim)
        4) cat => shape => (S,E,A, obs_dim + id_emb_dim)
        """
        S, E, A, obs_dim = obs.shape
        device = obs.device

        # 1) Flatten + apply pre_obs_transform
        flat_obs = obs.view(S*E*A, obs_dim)
        flat_trans = self.pre_obs_transform(flat_obs)  # => shape (S*E*A, ???)
        # re-reshape
        new_obs = flat_trans.view(S,E,A,-1)

        # 2) build agent_ids => shape => (S,E,A)
        agent_ids = torch.arange(A, device=device).view(1,1,A).expand(S,E,A)
        # 3) embed => (S,E,A, id_emb_dim)
        id_emb = self.id_encoder(agent_ids)

        # 4) cat => shape => (S,E,A, new_obs_dim + id_emb_dim)
        obs_with_id = torch.cat([new_obs, id_emb], dim=-1)
        return obs_with_id

    def _obs_with_id_and_rsa(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Common code path for [policy / value].
        1)  pre_obs_transform on obs
        2)  add agent ID => shape => (S,E,A, ??? + id_emb_dim)
        3)  pass that into g => (S,E,A,H)
        4)  flatten => RSA => final => (S,E,A,H)
        """
        # 1) => new obs with ID appended
        obs_with_id = self._add_id_emb(obs)

        # 2) pass to g => (S,E,A,H)
        obs_emb = self.g(obs_with_id)

        # 3) flatten => RSA => (S,E,A,H)
        S, E, A, hdim = obs_emb.shape
        flat = obs_emb.view(S*E, A, hdim)
        att  = self.rsa(flat)
        return att.view(S,E,A,-1)

    # ---------------------------------------------------------------------
    #            SPLIT: agent obs vs. groupmates obs+actions
    # ---------------------------------------------------------------------
    def split_agent_obs_groupmates_obs_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Splits the (S,E,A,...) array into:
          - agent_obs => (S,E,A,1, ???)
          - groupmates_obs => (S,E,A,(A-1), ???)
          - groupmates_actions => (S,E,A,(A-1), act_dim)
        """
        S,E,A,aug_dim = obs.shape
        act_dim = actions.shape[-1]

        agent_obs_list = []
        groupmates_obs_list = []
        groupmates_actions_list = []

        for i in range(A):
            # agent i's obs => (S,E,1,aug_dim)
            agent_i = obs[..., i, :].unsqueeze(2)
            agent_obs_list.append(agent_i)

            # groupmates => exclude i => (S,E,(A-1),aug_dim)
            g_obs = torch.cat([obs[..., :i, :], obs[..., i+1:, :]], dim=2).unsqueeze(2)
            g_act = torch.cat([actions[..., :i, :], actions[..., i+1:, :]], dim=2).unsqueeze(2)

            groupmates_obs_list.append(g_obs)
            groupmates_actions_list.append(g_act)

        agent_obs = torch.cat(agent_obs_list, dim=2).unsqueeze(3).contiguous()
        # => (S,E,A,1,aug_dim)
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        # => (S,E,A,(A-1),aug_dim)
        groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()
        # => (S,E,A,(A-1), act_dim)

        return agent_obs, groupmates_obs, groupmates_actions


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
