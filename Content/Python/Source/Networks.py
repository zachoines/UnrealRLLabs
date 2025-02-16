import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from typing import Dict, Any

# ------------------------------------------------------------------------
#  Basic Building Blocks
# ------------------------------------------------------------------------
class LinearNetwork(nn.Module):
    """
    A simple utility MLP block: Linear + optional activation + dropout.
    Used by state encoders, action encoders, etc.
    """
    def __init__(self, in_features: int, out_features: int,
                 dropout_rate=0.0, activation=True):
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
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class RSA(nn.Module):
    """
    Relational Self-Attention (RSA):
    Produces relational embeddings among agents.

    Input:  (Batch,Agents,H)
    Output: (Batch,Agents,H)
    """
    def __init__(self, embed_size: int, heads: int, dropout_rate=0.0):
        super(RSA, self).__init__()
        self.query_embed = self._linear_layer(embed_size, embed_size)
        self.key_embed   = self._linear_layer(embed_size, embed_size)
        self.value_embed = self._linear_layer(embed_size, embed_size)
        
        self.input_norm  = nn.LayerNorm(embed_size)
        self.output_norm = nn.LayerNorm(embed_size)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_size, heads, batch_first=True, dropout=dropout_rate
        )

    def _linear_layer(self, in_size: int, out_size: int) -> nn.Module:
        layer = nn.Linear(in_size, out_size)
        init.kaiming_normal_(layer.weight)
        nn.init.constant_(layer.bias, 0.0)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => shape (Batch, Agents, H)
        """
        x = self.input_norm(x)
        q = self.query_embed(x)
        k = self.key_embed(x)
        v = self.value_embed(x)
        out, _ = self.multihead_attn(q, k, v)
        out = x + out  # residual
        return self.output_norm(out)

class AgentIDPosEnc(nn.Module):
    """
    For each integer agent index i, produce a sinusoidal+linear embedding of size id_embed_dim.
    """
    def __init__(self, num_freqs: int = 8, id_embed_dim: int = 32):
        super().__init__()
        self.num_freqs   = num_freqs
        self.id_embed_dim= id_embed_dim
        self.linear = nn.Linear(2*num_freqs, id_embed_dim)
        init.kaiming_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)

    def sinusoidal_features(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        agent_ids => (S,E,NA) int => returns => (S,E,NA,2*num_freqs)
        """
        agent_ids_f = agent_ids.float()
        freq_exponents = torch.arange(self.num_freqs, device=agent_ids.device, dtype=torch.float32)
        scales = torch.pow(2.0, freq_exponents)  # (num_freqs,)

        # shape => (S,E,NA,1)
        agent_ids_f = agent_ids_f.unsqueeze(-1)
        # shape => (1,num_freqs)
        scales = scales.view(1, self.num_freqs)

        multiplied = agent_ids_f*scales  # => (S,E,NA,num_freqs)
        sines  = torch.sin(multiplied)
        cosines= torch.cos(multiplied)
        feats  = torch.cat([sines, cosines], dim=-1)  # => (S,E,NA,2*num_freqs)
        return feats

    def forward(self, agent_ids: torch.Tensor) -> torch.Tensor:
        """
        agent_ids => shape (S,E,NA)
        returns => (S,E,NA, id_embed_dim)
        """
        raw_feat = self.sinusoidal_features(agent_ids)
        S,E,NA,D = raw_feat.shape
        flat = raw_feat.contiguous().view(S*E*NA, D)
        out  = self.linear(flat)
        return out.view(S,E,NA,self.id_embed_dim)


# ------------------------------------------------------------------------
#  GeneralObsEncoder: merges "central"/"agent" (and ID if agent) => aggregator
# ------------------------------------------------------------------------

class GeneralObsEncoder(nn.Module):
    """
    A dictionary-based encoder for multiple keys: "central", "agent", etc.
    Then merges them + (optionally) agent ID embeddings => aggregator.
    Produces shape => (S,E,NA, aggregator_output_size).
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.encoder_cfgs           = config["encoders"]
        self.aggregator_output_size = config["aggregator_output_size"]

        # Build sub-encoders
        self.encoders        = nn.ModuleDict()
        self.sub_enc_outputs = {}
        sum_dims = 0
        has_agent = False

        for enc_info in self.encoder_cfgs:
            key       = enc_info["key"]
            net_class = enc_info["network"]
            params    = enc_info["params"]

            if net_class == "SpatialNetwork2D":
                sub_enc = SpatialNetwork2D(**params)
                out_dim = sub_enc.out_features
            elif net_class == "LinearNetwork":
                sub_enc = LinearNetwork(**params)
                out_dim = params["out_features"]
            else:
                raise ValueError(f"Unknown sub-encoder: {net_class}")

            self.encoders[key] = sub_enc
            self.sub_enc_outputs[key] = out_dim
            sum_dims += out_dim

            if key == "agent":
                has_agent = True

        # If there's an "agent" key, we also expect "agent_id_enc" in config
        self.agent_id_enc = None
        self.agent_id_dim = 0
        if has_agent:
            if "agent_id_enc" not in config:
                raise ValueError("agent_id_enc not found in config despite 'agent' key.")
            id_cfg = config["agent_id_enc"]
            self.agent_id_enc = AgentIDPosEnc(**id_cfg)
            self.agent_id_dim = id_cfg["id_embed_dim"]
            sum_dims         += self.agent_id_dim

        # aggregator linear
        self.aggregator_linear = LinearNetwork(sum_dims, self.aggregator_output_size)

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict: e.g. { "central":(S,E,cDim), "agent":(S,E,NA,aDim) }
        => returns => (S,E,NA, aggregator_output_size)
        """
        sub_outputs = []
        final_S, final_E, final_NA = 1,1,1  # placeholders

        # 1) Build sub-encoder outputs
        for key, sub_enc in self.encoders.items():
            x = obs_dict.get(key, None)
            if x is None:
                continue

            if key == "central":
                # shape => (S,E,cDim)
                S,E,cDim = x.shape
                b = S*E
                emb_2d = sub_enc(x.contiguous().view(b, cDim))  # => (b, out_dim)
                out_dim= self.sub_enc_outputs[key]
                emb_3d = emb_2d.view(S,E,out_dim)               # => (S,E,out_dim)
                emb_4d = emb_3d.unsqueeze(2)                    # => (S,E,1,out_dim)
                final_S, final_E, final_NA = S,E,1
                sub_outputs.append(emb_4d)

            elif key == "agent":
                # shape => (S,E,NA,aDim)
                S,E,NA,aDim = x.shape
                b = S*E*NA
                emb_2d = sub_enc(x.contiguous().view(b, aDim))  # => (b, out_dim)
                out_dim= self.sub_enc_outputs[key]
                emb_4d = emb_2d.view(S,E,NA,out_dim)            # => (S,E,NA,out_dim)
                final_S, final_E, final_NA = S,E,NA
                sub_outputs.append(emb_4d)

        # 2) If there's an agent => add agent ID embeddings
        if (self.agent_id_enc is not None) and ("agent" in obs_dict):
            # shape => (S,E,NA,aDim)
            agent_tensor = obs_dict["agent"]
            S,E,NA,_ = agent_tensor.shape
            agent_ids = torch.arange(NA, device=agent_tensor.device).view(1,1,NA)
            agent_ids = agent_ids.expand(S,E,NA)  # => (S,E,NA)
            id_4d = self.agent_id_enc(agent_ids)  # => (S,E,NA, id_embed_dim)
            sub_outputs.append(id_4d)

        # 3) If we only had "central" => final_NA=1; If we had "agent" => final_NA=NA
        # Expand each sub-output to match final_NA
        for i in range(len(sub_outputs)):
            so = sub_outputs[i]
            # shape => (S,E,xNA, dim)
            if so.shape[2] == 1 and final_NA > 1:
                # expand
                so = so.expand(so.shape[0], so.shape[1], final_NA, so.shape[-1])
            sub_outputs[i] = so

        if len(sub_outputs) > 0:
            combined = torch.cat(sub_outputs, dim=-1)  # => (S,E,NA, sum_dims)
        else:
            # empty => produce (1,1,1,0) => aggregator won't do anything
            combined = torch.zeros((1,1,1,0), device=next(self.parameters()).device)

        S2,E2,NA2,sum_dim = combined.shape
        flat = combined.view(S2*E2*NA2, sum_dim)
        agg_out_2d = self.aggregator_linear(flat)  # => (S2*E2*NA2, aggregator_output_size)
        return agg_out_2d.view(S2,E2,NA2,self.aggregator_output_size)


# ------------------------------------------------------------------------
#   StatesEncoder / StatesActionsEncoder
# ------------------------------------------------------------------------

class StatesEncoder(nn.Module):
    """
    e.g. g(...) in older code: takes (S,E,NA, aggregator_out_dim)
    => (S,E,NA,H)
    """
    def __init__(self, state_dim, output_size,
                 dropout_rate=0.0, activation=True):
        super(StatesEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim, output_size,
                                dropout_rate, activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class StatesActionsEncoder(nn.Module):
    """
    e.g. f(...) => for groupmates' obs+actions
    Input: obs => (S,E,A,(A-1),H)
           act => (S,E,A,(A-1), actDim)
    => output => (S,E,A,(A-1), H')
    """
    def __init__(self, state_dim, action_dim, output_size,
                 dropout_rate=0.0, activation=True):
        super(StatesActionsEncoder, self).__init__()
        self.fc = LinearNetwork(state_dim + action_dim, output_size,
                                dropout_rate, activation)

    def forward(self,
                observation: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Cat along last dim => pass MLP => shape same # of leading dims
        """
        x = torch.cat([observation, action], dim=-1)  # => (S,E,A,(A-1), state_dim+action_dim)
        return self.fc(x)


# ------------------------------------------------------------------------
#   ValueNetwork, SharedCritic, Policy
# ------------------------------------------------------------------------

class ValueNetwork(nn.Module):
    """
    A small MLP for producing a scalar value from a per-env or per-agent embedding.
    Input => (Batch, Features)
    Output => (Batch, 1)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_net(x)


class SharedCritic(nn.Module):
    """
    Has two MLP heads:
     1) value_head => V(s)
     2) baseline_head => Q_ψ for counterfactual baseline
    """
    def __init__(self, config):
        super(SharedCritic, self).__init__()
        net_cfg = config['agent']['params']['networks']['critic_network']
        self.value_head    = ValueNetwork(**net_cfg['value_head'])
        self.baseline_head = ValueNetwork(**net_cfg['baseline_head'])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def values(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (S,E,A,H). We average across agents => shape => (S,E,H)
        => pass value_head => => (S,E,1)
        """
        S,E,A,H = x.shape
        mean_emb = x.mean(dim=2)  # => (S,E,H)
        flat     = mean_emb.view(S*E, H)
        vals     = self.value_head(flat).view(S,E,1)
        return vals

    def baselines(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (S,E,A,A,H). Average across groupmates => shape => (S,E,A,H)
        => pass baseline_head => => (S,E,A,1)
        """
        S,E,A,A2,H = x.shape
        assert A == A2, "x must be shape (S,E,A,A,H)."
        mean_x = x.mean(dim=3)  # => (S,E,A,H)
        flat   = mean_x.view(S*E*A, H)
        base   = self.baseline_head(flat).view(S,E,A,1)
        return base


class DiscretePolicyNetwork(nn.Module):
    """
    For discrete or multi-discrete action spaces
    """
    def __init__(self, in_features: int, out_features, hidden_size: int):
        super().__init__()
        self.in_features  = in_features
        self.hidden_size  = hidden_size
        self.shared = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
        )

        # either single-cat or multi-cat
        if isinstance(out_features, int):
            self.head = nn.Linear(hidden_size, out_features)
        else:
            # multi-discrete
            self.branch_heads = nn.ModuleList([
                nn.Linear(hidden_size, branch_sz)
                for branch_sz in out_features
            ])
            self.out_is_multi = True

        # init
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
        returns => single-cat or list of cat logits
        """
        feats = self.shared(x)
        if hasattr(self, 'head'):
            return self.head(feats)
        else:
            # multi-discrete
            return [bh(feats) for bh in self.branch_heads]

class ContinuousPolicyNetwork(nn.Module):
    """
    Outputs unbounded mean + clamped log_std => Normal dist
    """
    def __init__(self, in_features: int, out_features: int,
                 hidden_size: int,
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

    def forward(self, x: torch.Tensor):
        feats   = self.shared(x)
        mean    = self.mean(feats)
        log_std = self.log_std_layer(feats)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std     = torch.exp(log_std)
        return mean, std

class QNetwork(nn.Module):
    """
    Optional Q-network for Q^\pi(s,a). Not actively used in MAPOCA code, but provided.
    Input => (batch, state_dim+action_dim)
    Output => (batch,1)
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        init.kaiming_normal_(self.fc1.weight)
        init.kaiming_normal_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.fc1(x))
        return self.fc2(x)


# ------------------------------------------------------------------------
#  New MultiAgentEmbeddingNetwork
# ------------------------------------------------------------------------

class MultiAgentEmbeddingNetwork(nn.Module):
    def __init__(self, net_cfg: Dict[str, Any]):
        super(MultiAgentEmbeddingNetwork, self).__init__()

        # 1) The dictionary based obs embedder with final aggregation at end
        self.general_obs_encoder = GeneralObsEncoder(net_cfg["general_obs_encoder"])

        # 2) For obs and obs+actions encoding
        self.f = StatesActionsEncoder(**net_cfg["obs_actions_encoder"])
        self.g = StatesEncoder(**net_cfg["obs_encoder"])

        # 3) common RSA for policy, value, baseline
        self.common_rsa = RSA(**net_cfg.get("common_rsa", net_cfg["RSA"]))
        self.baseline_rsa = RSA(**net_cfg.get("baseline_rsa", net_cfg["RSA"]))

    # --------------------------------------------------------------------
    #  Produce "base embedding" from raw dictionary of observarion
    # --------------------------------------------------------------------
    def get_base_embedding(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        obs_dict => { "central":(S,E,cDim), "agent":(S,E,NA,aDim) }
        => shape => (S,E,NA, aggregator_out_dim)
        """
        return self.general_obs_encoder(obs_dict)


    # --------------------------------------------------------------------
    #  Step 2c: embed_for_baseline
    # --------------------------------------------------------------------
    def get_common_and_baseline_embeddings(self,
                           common_emb: torch.Tensor,
                           actions: torch.Tensor=None) -> torch.Tensor:
        if actions != None:
            groupmates_emb, groupmates_actions = \
                self._split_agent_and_groupmates(common_emb, actions)


            agent_obs_emb = self.g(common_emb) # => shape (S,E,A,1,H)
            groupmates_obs_emb= self.f(groupmates_emb, groupmates_actions) # => (S,E,A,(A-1),H)
            
            # (S,E,A,H) and (S,E,A,A,H2)
            return agent_obs_emb, torch.cat([agent_obs_emb.unsqueeze(dim=3), groupmates_obs_emb], dim=3)
        else:
            agent_obs_emb = self.g(common_emb)
            return agent_obs_emb, None
    
    def attend_for_baselines(self, embeddings):
        return self._apply_attention(self.baseline_rsa, embeddings)
    
    def attend_for_common(self, embeddings):
        return self._apply_attention(self.common_rsa, embeddings)
    
    def _apply_attention(self, RSA, embeddings):
        # Unpack all but the last two dimensions into batch_dims:
        *batch_dims, NA, H = embeddings.shape
        
        # Flatten everything except the 'NA' and 'H' dimensions:
        embeddings_flat = embeddings.view(math.prod(batch_dims), NA, H)
        
        # Apply  RSA module:
        attended_flat = RSA(embeddings_flat)
        
        # Reshape the output back to the original dimensions:
        attended = attended_flat.view(*batch_dims, NA, H)
        
        return attended

    # --------------------------------------------------------------------
    #  Helper: _split_agent_and_groupmates
    # --------------------------------------------------------------------
    def _split_agent_and_groupmates(self,
                                    obs_4d: torch.Tensor,
                                    actions: torch.Tensor):
        """
        obs_4d => (S,E,A,H)
        actions=> (S,E,A, act_dim)  [assuming single vector per agent]
                  or (S,E,A, # branches) for discrete
        We want:
          agent_obs => (S,E,A,1,H)
          groupmates_obs => (S,E,A,A-1,H)
          groupmates_actions => (S,E,A,A-1, act_dim)
        """
        S,E,A,H = obs_4d.shape
        act_dim = actions.shape[-1]

        group_obs_list = []
        group_act_list = []

        for i in range(A):
            # groupmates => everything but agent i
            gm_obs = torch.cat([obs_4d[..., :i, :],
                                obs_4d[..., i+1:, :]], dim=2).unsqueeze(2)
            gm_act = torch.cat([actions[..., :i, :],
                                actions[..., i+1:, :]], dim=2).unsqueeze(2)

            group_obs_list.append(gm_obs)
            group_act_list.append(gm_act)

        # combine
        group_obs = torch.cat(group_obs_list, dim=2)               # => (S,E,A,A-1,H)
        group_act = torch.cat(group_act_list, dim=2)               # => (S,E,A,A-1,act_dim)

        return group_obs.contiguous(), group_act.contiguous()

class LN2d(nn.Module):
    """
    Applies LayerNorm across the channel dimension only.
    Shape: (B, C, H, W) -> LN -> same shape.
    """
    def __init__(self, num_channels):
        super().__init__()
        # We normalize over 'num_channels' only, ignoring H and W.
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x => (B, C, H, W)
        # 1) permute => (B, H, W, C)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        # 2) apply LN over the last dimension (channels)
        x = self.ln(x)
        # 3) permute back => (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x

class ResidualBlock(nn.Module):
    """
    Same as before; a simple residual block with LN2d (channel-only LN).
    """
    def __init__(self, channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.ln1   = LN2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.ln2   = LN2d(channels)

        init.kaiming_normal_(self.conv1.weight)
        nn.init.constant_(self.conv1.bias, 0.0)
        init.kaiming_normal_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.ln1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.ln2(out)

        out += residual
        out = F.relu(out)
        return out

class SpatialNetwork2D(nn.Module):
    """
    Updated network incorporating:
      - Coordinate channels
      - Residual blocks w/ LN2d (channel-only layer norm)
      - A final linear to choose output size
      - Reasonable defaults for 50x50 inputs

    Now LN2d won't mismatch shapes.
    """
    def __init__(
        self,
        h: int = 50,
        w: int = 50,
        in_channels: int = 1,
        base_channels: int = 16,
        num_blocks: int = 4,
        out_features: int = 128
    ):
        super().__init__()
        self.h = h
        self.w = w
        self.out_dim = out_features

        # +2 for the (row, col) coordinate channels
        self.initial_in_channels = in_channels + 2

        # Entry conv
        self.entry_conv = nn.Conv2d(self.initial_in_channels, base_channels, kernel_size=3, padding=1)
        init.kaiming_normal_(self.entry_conv.weight)
        nn.init.constant_(self.entry_conv.bias, 0.0)

        self.entry_ln = LN2d(base_channels)

        # Residual blocks (with occasional pooling)
        blocks = []
        current_channels = base_channels
        for i in range(num_blocks):
            blocks.append(ResidualBlock(current_channels, kernel_size=3, padding=1))
            # Pool every 2 blocks
            if (i + 1) % 2 == 0:
                blocks.append(nn.MaxPool2d(kernel_size=2))

        self.res_blocks = nn.Sequential(*blocks)

        # Calculate final spatial size after pooling
        pool_count = num_blocks // 2
        final_h = h // (2 ** pool_count)
        final_w = w // (2 ** pool_count)

        # Final linear layer
        self.final_fc = nn.Linear(current_channels * final_h * final_w, out_features)
        init.kaiming_normal_(self.final_fc.weight)
        nn.init.constant_(self.final_fc.bias, 0.0)

        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x => (B, 2500) for a 50x50 map
        1) reshape => (B,1,h,w)
        2) add coordinate channels => (B,3,h,w)
        3) entry conv + LN + ReLU
        4) pass residual blocks
        5) flatten + final FC => (B, out_features)
        """
        B, N = x.shape
        if N != self.h * self.w:
            raise ValueError(f"Expect input size {self.h*self.w}, got {N}.")

        # 1) reshape
        x_img = x.view(B, 1, self.h, self.w)

        # 2) coordinate channels
        device = x.device
        rows = torch.linspace(0, 1, self.h, device=device).view(1, 1, self.h, 1)
        cols = torch.linspace(0, 1, self.w, device=device).view(1, 1, 1, self.w)

        row_channel = rows.expand(B, 1, self.h, self.w)
        col_channel = cols.expand(B, 1, self.h, self.w)

        x_coord = torch.cat([x_img, row_channel, col_channel], dim=1).contiguous()  # => (B,3,h,w)

        # 3) entry conv => LN => ReLU
        out = self.entry_conv(x_coord)
        out = self.entry_ln(out)
        out = F.relu(out)

        # 4) residual blocks
        out = self.res_blocks(out)

        # 5) flatten => final fc
        B2, C2, H2, W2 = out.shape
        out_flat = out.contiguous().view(B2, C2 * H2 * W2)
        emb = self.final_fc(out_flat)

        return emb
