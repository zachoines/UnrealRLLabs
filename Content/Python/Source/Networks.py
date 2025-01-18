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

class SpatialNetwork2D(nn.Module):
    """
    Simple network that treats each batch entry as a flattened NxM 2D matrix.
    """
    def __init__(self, h, w, in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # We'll store h,w for reshaping
        self.h = h
        self.w = w
        self.out_channels = out_channels
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(),
        )

        for layer in self.conv:
            if isinstance(layer, nn.Conv2d):
                init.kaiming_normal_(layer.weight)

    def forward(self, x):
        """
        x: (batch, in_features) => interpret as (batch,1,h,w)
        => conv => flatten => (batch, out_channels*h*w)
        """
        b, n = x.shape
        # reshape to (b, 1, h, w)
        x_4d = x.view(b, 1, self.h, self.w)
        y = self.conv(x_4d)  # => (b, out_channels, h, w)
        return y.view(b, -1) # (b, out_channels*h*w)
    
class GeneralObsEncoder(nn.Module):
    """
    GeneralObsEncoder:
      - input_size: total dimension of the incoming obs vector
      - output_size: final aggregator FC's out_features
      - sub_encoders: a list of sub-encoder configs:
           {
             "name": str,
             "network": "LinearNetwork" or "SpatialNetwork2D" or ...
             "params": dict,
             "slice": [start,end],  # inclusive
             "out_features": int
           }
    """
    def __init__(self, config: dict):
        super().__init__()

        self.input_size  = config["input_size"]     # total incoming obs dim
        self.output_size = config["output_size"]    # final aggregator out dim
        self.sub_enc_cfgs = config["sub_encoders"]  # list of sub-encoder dicts

        # 1) Validate & sort slices
        self._check_slices_and_coverage()

        # 2) Build sub-encoders
        self.sub_encoders = nn.ModuleList()
        total_out = 0
        for sub_cfg in self.sub_enc_cfgs:
            name        = sub_cfg["name"]
            net_class   = sub_cfg["network"]
            params      = sub_cfg["params"]
            slc         = sub_cfg["slice"]
            out_dim     = sub_cfg["out_features"]
            start_idx, end_idx = slc
            slice_len   = (end_idx - start_idx + 1)

            # Instantiate
            sub_enc = self._build_sub_encoder(net_class, params)
            self.sub_encoders.append(sub_enc)

            total_out += out_dim

        # final aggregator: (total_out -> self.output_size)
        self.final_fc = nn.Linear(total_out, self.output_size)
        init.kaiming_normal_(self.final_fc.weight)
        nn.init.constant_(self.final_fc.bias, 0.0)


    def _check_slices_and_coverage(self):
        """
        Ensures sub_encoders' slices exactly cover [0..self.input_size-1] 
        with no gap or overlap, in ascending order.
        """
        # gather slices
        slices = []
        for sub_cfg in self.sub_enc_cfgs:
            sl = sub_cfg["slice"]
            if len(sl) != 2:
                raise ValueError(f"Invalid slice for sub-encoder {sub_cfg['name']}: {sl}")
            start, end = sl
            if not (0 <= start <= end < self.input_size):
                raise ValueError(f"Slice {sl} out of range for input_size={self.input_size}")
            slices.append((start, end))

        # sort by start
        slices.sort(key=lambda x: x[0])
        prev_end = -1
        for (st, en) in slices:
            if st != prev_end + 1 and prev_end != -1:
                raise ValueError(f"Gap/overlap in slices. Some sub-encoder starts at {st}, "
                                 f"previous ended at {prev_end}.")
            prev_end = en
        # final check: last sub-encoder must end at (input_size -1)
        if slices[-1][1] != self.input_size -1:
            raise ValueError(f"Coverage incomplete. Last slice ends at {slices[-1][1]}, "
                             f"but input_size-1= {self.input_size-1}.")


    def _build_sub_encoder(self, net_class_name: str, params: dict) -> nn.Module:
        """
        A small helper that instantiates the requested sub-encoder by name.
        We place them in the same file for simplicity.
        """
        # local import
        factory = {
            "LinearNetwork": LinearNetwork,
            "SpatialNetwork2D": SpatialNetwork2D,
        }
        if net_class_name not in factory:
            raise ValueError(f"Unknown sub-encoder class: {net_class_name}")
        cls = factory[net_class_name]
        return cls(**params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (batch, input_size)
        1) For each sub-encoder: slice => pass => shape => (batch, out_features)
        2) cat => final_fc => (batch, output_size)
        """
        b, dim = x.shape
        if dim != self.input_size:
            raise ValueError(f"GeneralObsEncoder expects input dim={self.input_size}, got {dim}")

        outs = []
        # each sub-encoder's slice => pass
        for sub_cfg, sub_enc in zip(self.sub_enc_cfgs, self.sub_encoders):
            start, end = sub_cfg["slice"]
            sub_x = x[:, start:end+1]  # inclusive
            out_i = sub_enc(sub_x)     # => (b, sub_cfg["out_features"])
            outs.append(out_i)

        cat_out = torch.cat(outs, dim=-1)  # => (b, sum_of_subencoder_out)
        return self.final_fc(cat_out)      # => (b, output_size)

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
            nn.Linear(hidden_size, hidden_size),
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
        gen_obs_cfg   = net_cfg["general_obs_encoder"]
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
        
        self.general_obs_encoder = GeneralObsEncoder(gen_obs_cfg)

    def embed_obs_for_policy(self, obs: torch.Tensor) -> torch.Tensor:
        """
        eqn(Policy):  π( RSA(g(o^i)) )
        1) Pass obs -> general_obs_encoder
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
         1) general_obs_encoder + agent IDs
         2) split => agent_obs_i, groupmates_obs_j, groupmates_actions_j
         3) g(agent_obs_i), f(groupmates_obs_j, groupmates_actions_j)
         4) cat => (S,E,A,A,H)
         5) RSA => (S,E,A,A,H)
        """
        # 1) general_obs_encoder + ID
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
        1) Flatten + pass obs -> general_obs_encoder
        2) agent_ids => (S,E,A)
        3) pass through self.id_encoder => (S,E,A,id_emb_dim)
        4) cat => shape => (S,E,A, new_obs_dim + id_emb_dim)
        """
        S, E, A, obs_dim = obs.shape
        device = obs.device

        # 1) Flatten + apply general_obs_encoder
        flat_obs = obs.view(S*E*A, obs_dim)
        flat_trans = self.general_obs_encoder(flat_obs)  # => shape (S*E*A, ???)
        # re-reshape
        new_obs = flat_trans.view(S, E, A, -1)

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
        1)  flatten + pass obs -> general_obs_encoder
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
        groupmates_obs = torch.cat(groupmates_obs_list, dim=2).contiguous()
        groupmates_actions = torch.cat(groupmates_actions_list, dim=2).contiguous()

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
