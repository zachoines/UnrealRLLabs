# NOTICE: This file includes modifications generated with the assistance of generative AI.
# Original code structure and logic by the project author.
from typing import Dict
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from Source.StateRecorder import StateRecorder
from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer, LinearValueScheduler
from Source.Networks import (
    MultiAgentEmbeddingNetwork,
    SharedCritic,
    DiscretePolicyNetwork,
    BetaPolicyNetwork
)

class MAPOCAAgent(Agent):
    """
    Multi-agent PPO (MAPOCA) style agent using an Attention-based SharedCritic.
    - Supports Beta or Discrete policy networks.
    - Clipped value & baseline (optional).
    - Multi-epoch mini-batch updates.
    - Counterfactual baseline calculation.
    """

    def __init__(self, config: Dict, device: torch.device):
        super().__init__(config, device)

        agent_cfg = config["agent"]["params"]
        train_cfg = config["train"]
        shape_cfg = config["environment"]["shape"]
        action_cfg = shape_cfg["action"]
        rec_cfg = config.get("StateRecorder", None)

        # Hyperparameters
        self.gamma = agent_cfg.get("gamma", 0.99)
        self.lmbda = agent_cfg.get("lambda", 0.95) # GAE lambda
        self.lr = agent_cfg.get("learning_rate", 3e-4)
        self.value_loss_coeff = agent_cfg.get("value_loss_coeff", 0.5)
        self.baseline_loss_coeff = agent_cfg.get("baseline_loss_coeff", 0.5)
        self.entropy_coeff = agent_cfg.get("entropy_coeff", 0.01)
        self.max_grad_norm = agent_cfg.get("max_grad_norm", 0.5)
        self.normalize_adv = agent_cfg.get("normalize_advantages", False)
        self.ppo_clip_range = agent_cfg.get("ppo_clip_range", 0.1)
        self.clipped_value_loss = agent_cfg.get("clipped_value_loss", False)
        self.value_clip_range = agent_cfg.get("value_clip_range", 0.2)

        # Training Params
        self.epochs = train_cfg.get("epochs", 4)
        self.mini_batch_size = train_cfg.get("mini_batch_size", 128)

        # Recording
        self.state_recorder = None
        if rec_cfg:
            self.state_recorder = StateRecorder(rec_cfg)

        # Action Space Determination
        self._determine_action_space(action_cfg)

        # Build Networks
        self.embedding_network = MultiAgentEmbeddingNetwork(
            agent_cfg["networks"]["MultiAgentEmbeddingNetwork"]
        ).to(device)
        self.shared_critic = SharedCritic(
            agent_cfg["networks"]["critic_network"]
        ).to(device)

        # --- Policy Network Creation (MODIFIED) ---
        pol_cfg = agent_cfg["networks"]["policy_network"]
        if self.action_space_type == "continuous":
            # Use BetaPolicyNetwork for continuous actions
            pol_cfg['out_features'] = self.action_dim
            # Ensure BetaPolicyNetwork specific params are handled correctly
            # (e.g., param_init_bias, min_concentration might be in pol_cfg)
            self.policy_net = BetaPolicyNetwork(
                **pol_cfg
            ).to(device)
            print("Using Beta Policy Network for Continuous Actions")
        elif self.action_space_type == "discrete":
            # Use DiscretePolicyNetwork for discrete actions
            discrete_pol_cfg = pol_cfg.copy() # Avoid modifying original dict
            discrete_pol_cfg['out_features'] = self.action_dim # Set correct output features
            self.policy_net = DiscretePolicyNetwork(
                **discrete_pol_cfg
            ).to(device)
            print("Using Discrete Policy Network")
        else:
             raise ValueError(f"Unsupported action_space_type: {self.action_space_type}")
        # --- End Policy Network Creation Modification ---

        # Optimizer & Schedulers
        self.optimizer = optim.AdamW(self.parameters(), lr=self.lr, eps=1e-7)
        lr_sched_cfg = agent_cfg.get("schedulers", {}).get("lr", None) # Use .get for safety
        self.lr_scheduler = None
        if lr_sched_cfg:
            self.lr_scheduler = LinearLR(self.optimizer, **lr_sched_cfg)
        ent_sched_cfg = agent_cfg.get("schedulers", {}).get("entropy_coeff", None)
        self.entropy_scheduler = None
        if ent_sched_cfg:
            self.entropy_scheduler = LinearValueScheduler(**ent_sched_cfg)
            self.entropy_coeff = self.entropy_scheduler.start_value
        policy_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("policy_clip", None)
        self.policy_clip_scheduler = None
        if policy_clip_sched_cfg:
             self.policy_clip_scheduler = LinearValueScheduler(**policy_clip_sched_cfg)
             self.ppo_clip_range = self.policy_clip_scheduler.start_value
        value_clip_sched_cfg = agent_cfg.get("schedulers", {}).get("value_clip", None)
        self.value_clip_scheduler = None
        if value_clip_sched_cfg:
             self.value_clip_scheduler = LinearValueScheduler(**value_clip_sched_cfg)
             self.value_clip_range = self.value_clip_scheduler.start_value
        grad_norm_sched_cfg = agent_cfg.get("schedulers", {}).get("max_grad_norm", None)
        self.max_grad_norm_scheduler = None
        if grad_norm_sched_cfg:
             self.max_grad_norm_scheduler = LinearValueScheduler(**grad_norm_sched_cfg)
             self.max_grad_norm = self.max_grad_norm_scheduler.start_value

        # Normalizers
        rewards_normalization_cfg = agent_cfg.get('rewards_normalizer', None)
        self.rewards_normalizer = None
        if rewards_normalization_cfg:
            self.rewards_normalizer = RunningMeanStdNormalizer(
                **rewards_normalization_cfg,
                device=self.device
            )

    def parameters(self):
        """ Returns parameters of all networks for the optimizer. """
        return (
            list(self.embedding_network.parameters())
            + list(self.shared_critic.parameters())
            + list(self.policy_net.parameters())
        )

    def _determine_action_space(self, action_cfg: Dict):
        """ Helper to parse action space config. """
        if "agent" in action_cfg:
            agent_action_cfg = action_cfg["agent"]
            if "discrete" in agent_action_cfg:
                self.action_space_type = "discrete"
                d_list = agent_action_cfg["discrete"]
                self.action_dim = [d["num_choices"] for d in d_list] if len(d_list) > 1 else d_list[0]["num_choices"]
            elif "continuous" in agent_action_cfg:
                self.action_space_type = "continuous"
                # Assuming continuous is a list of spec dicts, count them
                self.action_dim = len(agent_action_cfg["continuous"])
            else:
                raise ValueError("Missing discrete/continuous under action/agent.")
        elif "central" in action_cfg:
             # Handling for potential central action space (less common but possible)
             central_action_cfg = action_cfg["central"]
             if "discrete" in central_action_cfg:
                 self.action_space_type = "discrete"
                 d_list = central_action_cfg["discrete"]
                 self.action_dim = [item["num_choices"] for item in d_list] if len(d_list) > 1 else d_list[0]["num_choices"]
             elif "continuous" in central_action_cfg:
                 self.action_space_type = "continuous"
                 self.action_dim = len(central_action_cfg["continuous"])
             else:
                 raise ValueError("Missing discrete/continuous under action/central.")
        else:
            raise ValueError("No agent or central block found in action config.")
        print(f"Determined Action Space: Type={self.action_space_type}, Dim={self.action_dim}")


    @torch.no_grad()
    def get_actions(self, states: dict, dones: torch.Tensor, truncs: torch.Tensor, eval: bool=False, record: bool=True):
        """ Computes actions based on current policy and state observations. """
        base_emb, _ = self.embedding_network.get_base_embedding(states)
        # Determine shape dynamically
        input_shape = base_emb.shape
        H = input_shape[-1]
        NA = input_shape[-2]
        leading_dims = input_shape[:-2] # e.g., (S, E) or just (S,) or ()
        total_batch_size = int(np.prod(leading_dims)) if leading_dims else 1

        emb_flat = base_emb.reshape(total_batch_size * NA, H)
        # Get actions, log_probs, entropies from policy network
        actions_flat, lp_flat, ent_flat = self.policy_net.get_actions(emb_flat, eval=eval)

        # Reshape outputs back to original batch structure + Agent dim
        log_probs = lp_flat.reshape(*leading_dims, NA)
        entropies = ent_flat.reshape(*leading_dims, NA)

        # Reshape actions based on action space type
        if self.action_space_type == "continuous":
            act_dim = actions_flat.shape[-1] # Should match self.action_dim
            actions = actions_flat.reshape(*leading_dims, NA, act_dim)
        else: # Discrete
            # Handle single discrete vs multi-discrete cases
            if actions_flat.dim() == 1: # Single discrete branch flattened
                actions = actions_flat.reshape(*leading_dims, NA, 1) # Add action dim
            else: # Multi-discrete branches already have last dim
                num_branches = actions_flat.shape[-1]
                actions = actions_flat.reshape(*leading_dims, NA, num_branches)

        # Optional state recording
        if record and self.state_recorder is not None and len(leading_dims) >= 2: # Need at least Step and Env dims
             env_dim_size = leading_dims[1] # Assuming (S, E, ...) structure
             if env_dim_size > 1: # Avoid recording if only one env instance (e.g., during eval)
                 try:
                      if "central" in states and states["central"].dim() >= 2: # Check central state exists and has at least S, E dims
                           # Record state from the second environment instance (index 1) of the first step (index 0)
                           c_state_to_record = states["central"][0, 1, ...].cpu().numpy() # Use ellipsis for remaining dims
                           # Ensure it's flattened correctly if needed by recorder
                           self.state_recorder.record_frame(c_state_to_record.flatten())
                 except IndexError:
                      print(f"Warning: Could not record state. State shape: {states.get('central', torch.empty(0)).shape}")

        return actions, (log_probs, entropies) # Return tuple for consistency

    def update(self,
           states: dict,
           next_states: dict,
           actions: torch.Tensor,
           rewards: torch.Tensor,
           dones: torch.Tensor,
           truncs: torch.Tensor):
        """ Performs a PPO update step using collected experience. """
        # Determine batch dimensions (NS=NumSteps, NE=NumEnvs, NA=NumAgents)
        # Handle cases where only central or only agent state is present
        NA = 1 # Default for central-only
        if "agent" in states and states["agent"] is not None:
            NS, NE, NA, _ = states["agent"].shape
        elif "central" in states and states["central"] is not None:
            # Central state might be (NS, NE, C_obs) or (NS, NE, C, H, W) -> flatten later
             NS, NE = states["central"].shape[:2]
             # NA remains 1 if only central state is present
        else:
            raise ValueError("Cannot determine batch dimensions from states dictionary.")

        rewards_raw_mean = rewards.mean().item()
        if self.rewards_normalizer:
            self.rewards_normalizer.update(rewards)
            rewards = self.rewards_normalizer.normalize(rewards)
        rewards_norm_mean = rewards.mean().item()

        # --- Data Gathering (using old policy/value parameters) ---
        with torch.no_grad():
            # Compute embeddings for current and next states
            base_emb, _ = self.embedding_network.get_base_embedding(states)        # (NS, NE, NA, H)
            next_base_emb, _ = self.embedding_network.get_base_embedding(next_states) # (NS, NE, NA, H)

            # Compute embeddings needed for baseline calculation
            # Requires state embeddings and actions
            baseline_emb = self.embedding_network.get_baseline_embeddings(base_emb, actions) # (NS, NE, NA, SeqLen, H)

            # Get value estimates from critic
            old_vals = self.shared_critic.values(base_emb)      # (NS, NE, 1)
            old_bases = self.shared_critic.baselines(baseline_emb) # (NS, NE, NA, 1)
            next_vals = self.shared_critic.values(next_base_emb) # (NS, NE, 1)

            # Recompute log probabilities of actions taken, using old policy parameters
            old_lp, _ = self._recompute_log_probs(base_emb, actions) # (NS, NE, NA)

            # Compute GAE returns
            returns = self._compute_td_lambda_returns(rewards, old_vals, next_vals, dones, truncs) # (NS, NE, 1)

        # --- Prepare for PPO Update Loop ---
        num_samples = NS * NE # Total number of (state, action, ...) tuples across steps and envs
        idxes = np.arange(num_samples)
        log_keys = ["Policy Loss", "Value Loss", "Baseline Loss", "Entropy", "Grad Norm",
                    "Advantage Raw Mean", "Advantage Raw Std", "Advantage Final Mean",
                    "Returns Mean", "Logprob Mean", "Logprob Min", "Logprob Max"]
        batch_logs = {k: [] for k in log_keys}
        grad_norms_count = 0

        # --- PPO Training Loop (Multiple Epochs, Mini-batches) ---
        for epoch in range(self.epochs):
            np.random.shuffle(idxes)
            num_minibatches = (num_samples + self.mini_batch_size - 1) // self.mini_batch_size

            for mb_idx in range(num_minibatches):
                start = mb_idx * self.mini_batch_size
                end = min((mb_idx + 1) * self.mini_batch_size, num_samples)
                mb_indices = idxes[start:end]
                MB = len(mb_indices) # Actual mini-batch size

                # Map flat indices to (step, env) indices
                step_indices = torch.tensor(mb_indices // NE, device=self.device, dtype=torch.long)
                env_indices = torch.tensor(mb_indices % NE, device=self.device, dtype=torch.long)

                # Slice data to get mini-batch samples
                st_mb = self._slice_state_dict_batch(states, step_indices, env_indices) # Dict with shapes (MB, ...)
                act_mb = actions[step_indices, env_indices]  # Shape (MB, NA, ActionDim)
                ret_mb = returns[step_indices, env_indices]  # Shape (MB, 1)
                olp_mb = old_lp[step_indices, env_indices]   # Shape (MB, NA)
                oval_mb = old_vals[step_indices, env_indices] # Shape (MB, 1)
                obase_mb = old_bases[step_indices, env_indices] # Shape (MB, NA, 1)

                # --- Forward Pass with Current Policy/Value Parameters ---
                base_emb_mb, _ = self.embedding_network.get_base_embedding(st_mb) # (MB, NA, H)
                baseline_emb_mb = self.embedding_network.get_baseline_embeddings(base_emb_mb, act_mb) # (MB, NA, SeqLen, H)

                # Recompute log probs and entropy with current policy
                new_lp_mb, ent_mb = self._recompute_log_probs(base_emb_mb, act_mb) # (MB, NA), (MB, NA)

                # Log stats of new log probs
                lp_detached = new_lp_mb.detach()
                batch_logs["Logprob Mean"].append(lp_detached.mean().item())
                batch_logs["Logprob Min"].append(lp_detached.min().item())
                batch_logs["Logprob Max"].append(lp_detached.max().item())

                # Compute new value and baseline estimates
                new_vals_mb = self.shared_critic.values(base_emb_mb) # (MB, 1)
                new_base_mb = self.shared_critic.baselines(baseline_emb_mb) # (MB, NA, 1)

                # --- Advantage Calculation (Per Agent) ---
                # Expand returns to match baseline shape: (MB, 1) -> (MB, NA, 1)
                ret_mb_exp = ret_mb.unsqueeze(1).expand(-1, NA, 1)
                adv_mb = ret_mb_exp - new_base_mb # Advantage: A(s,a) = R - B(s,a); Shape (MB, NA, 1)
                adv_mb_squeezed = adv_mb.squeeze(-1) # Remove trailing dim: (MB, NA)

                # Log raw advantage stats
                adv_raw_mean = adv_mb_squeezed.mean().item()
                adv_raw_std = adv_mb_squeezed.std().item()
                batch_logs["Advantage Raw Mean"].append(adv_raw_mean)
                batch_logs["Advantage Raw Std"].append(adv_raw_std)

                # Normalize advantages if configured
                if self.normalize_adv:
                    adv_mb_squeezed = (adv_mb_squeezed - adv_mb_squeezed.mean()) / (adv_mb_squeezed.std() + 1e-8)

                # Log final (potentially normalized) advantage mean
                batch_logs["Advantage Final Mean"].append(adv_mb_squeezed.mean().item())
                detached_adv = adv_mb_squeezed.detach() # Detach advantages before policy loss

                # --- Loss Calculation ---
                # Policy Loss (PPO Clip)
                pol_loss = self._ppo_clip_loss(new_lp_mb, olp_mb, detached_adv, self.ppo_clip_range)

                # Value Loss (Centralized V(s))
                if self.clipped_value_loss:
                    val_loss = self._clipped_value_loss(oval_mb, new_vals_mb, ret_mb, self.value_clip_range)
                else:
                    val_loss = F.mse_loss(new_vals_mb, ret_mb)

                # Baseline Loss (Per-agent B(s,a))
                if self.clipped_value_loss:
                    # Use clipped loss for baseline as well if enabled
                    base_loss = self._clipped_value_loss(obase_mb, new_base_mb, ret_mb_exp, self.value_clip_range)
                else:
                    base_loss = F.mse_loss(new_base_mb, ret_mb_exp)

                # Entropy Loss (Negative mean entropy across agents and batch)
                ent_mean = ent_mb.mean()
                entropy_loss = -ent_mean # We want to maximize entropy

                # Total Loss
                total_loss = (pol_loss +
                              self.value_loss_coeff * val_loss +
                              self.baseline_loss_coeff * base_loss + # Add entropy loss (minimize negative entropy)
                              self.entropy_coeff * entropy_loss)

                # --- Optimization Step ---
                self.optimizer.zero_grad()
                total_loss.backward()
                # Clip gradients
                gn = clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # --- Logging Mini-batch Stats ---
                batch_logs["Policy Loss"].append(pol_loss.item())
                batch_logs["Value Loss"].append(val_loss.item())
                batch_logs["Baseline Loss"].append(base_loss.item())
                batch_logs["Entropy"].append(ent_mean.item()) # Log positive entropy
                batch_logs["Grad Norm"].append(gn.item() if torch.is_tensor(gn) else gn)
                batch_logs["Returns Mean"].append(ret_mb.mean().item())
                grad_norms_count += 1

            # --- End of Mini-batch Loop ---

            # --- Scheduler Steps (End of Epoch) ---
            current_lr = self._get_avg_lr(self.optimizer) # Get LR before scheduler step
            if self.lr_scheduler: self.lr_scheduler.step()
            if self.entropy_scheduler: self.entropy_coeff = self.entropy_scheduler.step()
            if self.policy_clip_scheduler: self.ppo_clip_range = self.policy_clip_scheduler.step()
            if self.value_clip_scheduler: self.value_clip_range = self.value_clip_scheduler.step()
            if self.max_grad_norm_scheduler: self.max_grad_norm = self.max_grad_norm_scheduler.step()
        # --- End of Epoch Loop ---

        # --- Final Logging (Averaged over all mini-batches) ---
        final_logs = {k: np.mean(v) for k, v in batch_logs.items() if v} # Calculate mean for logged stats
        final_logs["Entropy coeff"] = self.entropy_coeff # Log final scheduler values
        final_logs["Learning Rate"] = current_lr
        final_logs["PPO Clip Range"] = self.ppo_clip_range
        final_logs["Max Grad Norm"] = self.max_grad_norm
        final_logs["Raw Reward Mean"] = rewards_raw_mean # Log raw reward mean for context
        final_logs["Norm Reward Mean"] = rewards_norm_mean # Log normalized reward mean

        return final_logs

    # ============================================================
    # Helper Methods (Mostly Unchanged)
    # ============================================================

    def _compute_td_lambda_returns(self, rewards, values, next_values, dones, truncs):
        """ Computes TD(Lambda) returns (GAE targets). """
        # rewards shape: (NS, NE, 1), values/next_values: (NS, NE, 1)
        NS, NE, _ = rewards.shape
        returns = torch.zeros_like(rewards)
        gae = torch.zeros(NE, 1, device=rewards.device) # Last GAE value per environment
        dones = dones.float()
        truncs = truncs.float() # Ensure float for multiplication

        for t in reversed(range(NS)):
            # If t is the last step (T-1), next value is from next_values buffer
            # Otherwise, next value is the value estimate at t+1 from the 'values' buffer
            v_next = next_values[t] # Value estimate of state s_{t+1}
            mask = (1.0 - dones[t]) * (1.0 - truncs[t]) # Don't bootstrap if terminal
            v_next_masked = v_next * mask

            # TD Residual: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + self.gamma * v_next_masked - values[t]

            # GAE: A_t = delta_t + gamma * lambda * A_{t+1} * mask_{t+1}
            # We iterate backwards, so gae holds A_{t+1}
            gae = delta + self.gamma * self.lmbda * mask * gae

            # Return R_t = A_t + V(s_t)
            returns[t] = gae + values[t]
        return returns

    def _ppo_clip_loss(self, new_lp, old_lp, advantages, clip_range):
        """ Computes the PPO clipped surrogate objective loss. """
        # new_lp, old_lp shape: (MB, NA)
        # advantages shape: (MB, NA)
        log_ratio = new_lp - old_lp
        # Clamp log_ratio for numerical stability (optional but recommended)
        log_ratio = torch.clamp(log_ratio, -10.0, 10.0)
        ratio = torch.exp(log_ratio)

        surr1 = ratio * advantages
        # Clip the ratio
        surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
        # Take the minimum (pessimistic bound) and negate for gradient ascent (maximize objective)
        policy_loss = -torch.min(surr1, surr2).mean() # Mean over batch and agents
        return policy_loss

    def _clipped_value_loss(self, old_values, new_values, targets, clip_range):
        """ Computes the PPO clipped value loss. """
        # old_values shape: (MB, 1) or (MB, NA, 1)
        # new_values shape: same as old_values
        # targets shape: (MB, 1) or (MB, NA, 1) - should match old_values

        # Clip new values based on old values
        values_clipped = old_values + torch.clamp(new_values - old_values, -clip_range, clip_range)
        # Calculate MSE loss for both clipped and unclipped values
        mse_unclipped = F.mse_loss(new_values, targets, reduction='none')
        mse_clipped = F.mse_loss(values_clipped, targets, reduction='none')
        # Take the maximum of the two losses for each element
        value_loss = torch.max(mse_unclipped, mse_clipped).mean() # Mean over all elements
        return value_loss

    def _recompute_log_probs(self, base_embeddings: torch.Tensor, actions: torch.Tensor):
        """ Recomputes log probs and entropies using the current policy network. """
        input_shape = base_embeddings.shape
        H = input_shape[-1] # Embedding dim
        NA = input_shape[-2] # Num agents
        leading_dims = input_shape[:-2] # Batch dimensions (e.g., NS, NE or just MB)
        total_batch_size = int(np.prod(leading_dims)) if leading_dims else 1

        # Reshape embeddings for the policy network: (TotalBatch * NA, H)
        emb_flat = base_embeddings.reshape(total_batch_size * NA, H)

        # Reshape actions: (TotalBatch * NA, ActionDim or NumBranches)
        # Determine action shape based on policy type
        if self.action_space_type == "continuous":
             action_last_dims = actions.shape[len(leading_dims)+1:] # Get dimensions after (..., NA)
             act_flat = actions.reshape(total_batch_size * NA, *action_last_dims)
        else: # Discrete
             # Handle single vs multi-discrete shaping if necessary
             if actions.shape[-1] == 1 and not isinstance(self.action_dim, list): # Single discrete squeezed
                 act_flat = actions.reshape(total_batch_size * NA) # Needs to be (TotalBatch * NA,)
             else: # Multi-discrete or single discrete kept last dim
                 action_last_dims = actions.shape[len(leading_dims)+1:]
                 act_flat = actions.reshape(total_batch_size * NA, *action_last_dims)


        # Call the policy network's recompute method
        lp_flat, ent_flat = self.policy_net.recompute_log_probs(emb_flat, act_flat)

        # Reshape results back to original batch structure + Agent dim
        log_probs = lp_flat.reshape(*leading_dims, NA)
        entropies = ent_flat.reshape(*leading_dims, NA)
        return log_probs, entropies

    def _slice_state_dict_batch(self, states_dict: dict, step_indices, env_indices):
         """ Slices state dict using step and env indices for mini-batch creation. """
         new_dict = {}
         for k, v in states_dict.items():
             if v is not None:
                 # Ensure tensor has at least 2 dimensions (step, env) for slicing
                 if v.dim() >= 2:
                     try:
                         # Slice using the provided step and environment indices
                         new_dict[k] = v[step_indices, env_indices].contiguous()
                     except IndexError as e:
                         print(f"Error slicing state '{k}' with shape {v.shape} using indices: {e}")
                         raise e # Re-raise after logging
                 else:
                     # Handle cases where state might be 1D or scalar (less common for batch data)
                     print(f"Warning: State '{k}' shape {v.shape} < 2D, cannot perform standard (step, env) slicing.")
                     # Decide how to handle this case, e.g., skip, try different slicing, or raise error
                     new_dict[k] = None # Skipping for now
             else:
                 new_dict[k] = None # Keep None values as None
         return new_dict

    def _get_avg_lr(self, optimizer) -> float:
        """ Gets the average learning rate from optimizer param groups. """
        lr_sum, count = 0.0, 0
        for pg in optimizer.param_groups:
            lr_sum += pg.get('lr', 0.0) # Use .get for safety
            count += 1
        return lr_sum / max(count, 1) # Avoid division by zero

    # _accumulate_per_layer_grad_norms remains unchanged
    def _accumulate_per_layer_grad_norms(self, grad_norms_sum_per_layer: dict):
        """ Accumulates L2 norm of gradients for each named parameter. """
        if not hasattr(self, '_param_names_with_grads'):
             # Cache the names of parameters that require gradients
             self._param_names_with_grads = {name for name, param in self.named_parameters() if param.requires_grad}
        for name, param in self.named_parameters():
            # Check if the parameter requires grad and has a computed gradient
            if name in self._param_names_with_grads and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item() # Calculate L2 norm
                # Accumulate the norm, initializing if key doesn't exist
                grad_norms_sum_per_layer[name] = grad_norms_sum_per_layer.get(name, 0.0) + grad_norm