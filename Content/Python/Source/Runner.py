# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import win32event

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer 
from Source.Environment import EnvCommunicationInterface, EventType
# Assuming StateRecorder is in the same directory or accessible via Source.
from Source.StateRecorder import StateRecorder 

# -----------------------------------------------------------------------------
class TrajectorySegment:
    """Accumulates one variable‑length segment until cut or padded."""

    def __init__(self, num_agents_cfg: int, device: torch.device, pad: bool, max_len: int):
        self.device = device
        self.num_agents_cfg = num_agents_cfg
        self.enable_padding = pad
        self.max_segment_length = max_len
        self.true_sequence_length = 0
        # storages
        self.observations: List[Dict[str, Any]] = [] 
        self.next_observations: List[Dict[str, Any]] = [] 
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.truncs: List[torch.Tensor] = []
        # newly stored values required by MA-POCA update
        self.log_probs: List[torch.Tensor] = []
        self.values: List[torch.Tensor] = []
        self.baselines: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self.returns: List[torch.Tensor] = []
        self.initial_hidden_state: Optional[torch.Tensor] = None

    def is_full(self) -> bool:
        return self.enable_padding and self.true_sequence_length >= self.max_segment_length

    def add_step(self, obs: Dict[str, Any], act: torch.Tensor, rew: torch.Tensor,
                 next_obs: Dict[str, Any], done: torch.Tensor, trunc: torch.Tensor,
                 log_prob: Optional[torch.Tensor] = None, value: Optional[torch.Tensor] = None,
                 baseline: Optional[torch.Tensor] = None, entropy: Optional[torch.Tensor] = None,
                 ret: Optional[torch.Tensor] = None):
        if self.is_full():
            print("RLRunner warn: add_step called on full segment")
            return
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(done)
        self.truncs.append(trunc)
        if log_prob is not None: self.log_probs.append(log_prob)
        if value is not None: self.values.append(value)
        if baseline is not None: self.baselines.append(baseline)
        if entropy is not None: self.entropies.append(entropy)
        if ret is not None: self.returns.append(ret)
        self.true_sequence_length += 1

    def set_returns(self, returns: List[torch.Tensor]):
        """Assign calculated returns to this segment."""
        self.returns = returns

    def tensors_for_collation(self) -> Tuple[
        List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor],
        List[Dict[str, Any]], List[torch.Tensor], List[torch.Tensor],
        List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
        Optional[torch.Tensor], int
    ]:
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.dones,
            self.truncs,
            self.log_probs,
            self.values,
            self.baselines,
            self.entropies,
            self.returns,
            self.initial_hidden_state,
            self.true_sequence_length,
        )

# -----------------------------------------------------------------------------
class RLRunner:
    """Handles UE→Python experience pipeline and agent updates."""

    def __init__(self, agent: Agent, agentComm: EnvCommunicationInterface, cfg: Dict):
        self.agent, self.agentComm, self.config = agent, agentComm, cfg
        self.device = agentComm.device 

        trn_cfg = cfg["train"]
        ag_cfg = cfg["agent"]["params"]
        env_shape = cfg["environment"]["shape"]
        env_params = cfg["environment"]["params"]
        
        # State Recorder Initialization
        recorder_config = cfg.get("StateRecorder", None) # Get StateRecorder config block
        if cfg.get("StateRecorder_disabled", None) is not None : # Check if "StateRecorder_disabled" key exists
             print("StateRecorder is explicitly disabled via 'StateRecorder_disabled' key in config.")
             self.state_recorder = None
        elif recorder_config:
            print("Initializing StateRecorder...")
            self.state_recorder = StateRecorder(recorder_config)
        else:
            print("StateRecorder config not found or disabled.")
            self.state_recorder = None


        self.sequence_length = trn_cfg.get("sequence_length", 64)
        self.pad_trajectories = trn_cfg.get("pad_trajectories", True)
        self.timesteps_from_ue_batch = trn_cfg.get("batch_size", 256) 
        self.num_envs = trn_cfg.get("num_environments", 1)
        self.save_freq = trn_cfg.get("saveFrequency", 50)

        self.enable_memory = ag_cfg.get("enable_memory", False)
        mem_cfg = ag_cfg.get(ag_cfg.get("memory_type", "gru"), {})
        self.memory_hidden_size = mem_cfg.get("hidden_size", 128)
        self.memory_layers = mem_cfg.get("num_layers", 1)

        self.num_agents_cfg = env_params.get("MaxAgents", 1)
        if "agent" in env_shape.get("state", {}): 
            self.num_agents_cfg = env_shape["state"]["agent"].get("max", self.num_agents_cfg)

        self.current_memory_hidden_states = (
            torch.zeros(
                self.num_envs,
                self.num_agents_cfg,
                self.memory_layers,
                self.memory_hidden_size,
                device=self.device,
            )
            if self.enable_memory
            else None
        )

        self.current_segments = [
            TrajectorySegment(self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length)
            for _ in range(self.num_envs)
        ]
        if self.enable_memory and self.current_memory_hidden_states is not None:
            for e in range(self.num_envs):
                self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        # lists of segments that form the currently running episode for each environment
        self.current_episode_segments: List[List[TrajectorySegment]] = [[] for _ in range(self.num_envs)]

        # segments that have completed and have computed returns, ready for update
        self.completed_segments: List[List[TrajectorySegment]] = [[] for _ in range(self.num_envs)]

        # per-environment pending step buffers storing info collected at get_actions time
        self.pending_steps: List[List[Dict[str, Any]]] = [[] for _ in range(self.num_envs)]

        norm_cfg = trn_cfg.get("states_normalizer", None)
        self.state_normalizer = RunningMeanStdNormalizer(**norm_cfg, device=self.device, dtype=torch.float32) if norm_cfg else None

        self.writer = SummaryWriter()
        self.update_idx = 0
        print(f"RLRunner initialised (envs: {self.num_envs}, device: {self.device})")

    def _finalize_episode(self, env_index: int):
        """Compute returns for all segments of a finished episode."""
        episode_segments = self.current_episode_segments[env_index]
        if not episode_segments:
            return

        # gather episode level tensors
        rewards = []
        values = []
        dones = []
        truncs = []
        for seg in episode_segments:
            rewards.extend(seg.rewards)
            values.extend(seg.values)
            dones.extend(seg.dones)
            truncs.extend(seg.truncs)

        r_t = torch.stack(rewards, dim=0).unsqueeze(0)  # (1,T,1)
        v_t = torch.stack(values, dim=0).unsqueeze(0)
        d_t = torch.stack(dones, dim=0).unsqueeze(0)
        tr_t = torch.stack(truncs, dim=0).unsqueeze(0)

        with torch.no_grad():
            returns = self.agent.compute_returns(r_t, v_t, d_t, tr_t)
        returns_ep = returns.squeeze(0)
        step_idx = 0
        for seg in episode_segments:
            seg_returns = returns_ep[step_idx : step_idx + seg.true_sequence_length]
            seg.set_returns([r for r in seg_returns])
            step_idx += seg.true_sequence_length
            self.completed_segments[env_index].append(seg)

        episode_segments.clear()

    def _finalize_rollout(self, env_index: int, bootstrap_value: torch.Tensor):
        """Finalize an unfinished rollout by bootstrapping the last state."""
        segments = self.current_episode_segments[env_index]
        if self.current_segments[env_index].true_sequence_length > 0:
            segments.append(self.current_segments[env_index])
        if not segments:
            return

        rewards: List[torch.Tensor] = []
        values: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        truncs: List[torch.Tensor] = []
        for seg in segments:
            rewards.extend(seg.rewards)
            values.extend(seg.values)
            dones.extend(seg.dones)
            truncs.extend(seg.truncs)

        r_t = torch.stack(rewards, dim=0).unsqueeze(0)
        v_t = torch.stack(values, dim=0).unsqueeze(0)
        d_t = torch.stack(dones, dim=0).unsqueeze(0)
        tr_t = torch.stack(truncs, dim=0).unsqueeze(0)

        with torch.no_grad():
            returns = self.agent.compute_bootstrapped_returns(
                r_t, v_t, d_t, tr_t, bootstrap_value.view(1, 1)
            )
        returns_ep = returns.squeeze(0)

        step_idx = 0
        for seg in segments:
            seg_returns = returns_ep[step_idx : step_idx + seg.true_sequence_length]
            seg.set_returns([r for r in seg_returns])
            step_idx += seg.true_sequence_length
            self.completed_segments[env_index].append(seg)

        segments.clear()
        self.current_segments[env_index] = TrajectorySegment(
            self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
        )
        if self.enable_memory and self.current_memory_hidden_states is not None:
            self.current_segments[env_index].initial_hidden_state = (
                self.current_memory_hidden_states[env_index].clone()
            )


    def start(self):
        while True:
            evt = self.agentComm.wait_for_event()
            if evt == EventType.GET_ACTIONS:
                self._handle_get_actions()
            elif evt == EventType.UPDATE:
                self._handle_update()
            else:
                print(f"RLRunner: Received unknown event type {evt}. Assuming shutdown or error.")
                break # Break the loop on unknown event
        self.end()


    def _handle_get_actions(self):
        states_from_comm, dones_from_comm, truncs_from_comm, needs_action = self.agentComm.get_states()
        
        if (
            not states_from_comm
            or "central" not in states_from_comm
            or not states_from_comm.get("central")
            or "agent" not in states_from_comm
            or states_from_comm.get("agent") is None
        ):
            # This can happen if initial shared memory read is empty or parsing fails in agentComm
            print("RLRunner: Received empty or invalid states_from_comm in _handle_get_actions. Skipping.")
            # Still need to signal UE to avoid deadlock if it's waiting for actions
            # Assuming a dummy action send might be needed or a more robust error signal to UE
            # For now, let's try to send zero actions if action_dim is known from Agent
            if hasattr(self.agent, 'total_action_dim_per_agent') and self.agent.total_action_dim_per_agent > 0:
                num_total_actions_per_env = self.num_agents_cfg * self.agent.total_action_dim_per_agent
                dummy_actions = torch.zeros((self.num_envs, num_total_actions_per_env), device=self.device)
                self.agentComm.send_actions(dummy_actions)
            else: # Fallback if action dim isn't readily available
                self.agentComm.send_actions(torch.empty(0, device=self.device)) # Or handle error more gracefully
            return

        current_states_dict: Dict[str, Any] = {}
        has_central_state = "central" in states_from_comm and states_from_comm["central"]
        has_agent_state = "agent" in states_from_comm and states_from_comm["agent"] is not None

        if not has_central_state and not has_agent_state:
            raise ValueError("RLRunner: states_from_comm must contain at least 'central' or 'agent' key with data.")

        if has_central_state:
            current_states_dict["central"] = {
                k: v.squeeze(0) for k, v in states_from_comm["central"].items()
            }
        if has_agent_state:
            current_states_dict["agent"] = states_from_comm["agent"].squeeze(0)

        dones = dones_from_comm.squeeze(0).squeeze(-1)
        truncs = truncs_from_comm.squeeze(0).squeeze(-1)
        needs_mask = needs_action.squeeze(0).squeeze(-1)

        if self.enable_memory and self.current_memory_hidden_states is not None:
            for e in range(self.num_envs):
                if needs_mask[e] > 0.5 and ((dones[e] > 0.5) or (truncs[e] > 0.5)):
                    # reset memory for new episode but postpone segment finalization
                    self.current_memory_hidden_states[e].zero_()
        
        states_for_agent_action = current_states_dict
        if self.state_normalizer:
            # --- Update Step ---
            if has_central_state:
                central_float_states_to_update = {
                    k: v for k, v in current_states_dict["central"].items() if torch.is_floating_point(v)
                }
                if central_float_states_to_update:
                    self.state_normalizer.update(central_float_states_to_update)
            
            if has_agent_state and torch.is_floating_point(current_states_dict["agent"]):
                self.state_normalizer.update({"agent": current_states_dict["agent"]})
            
            # --- Normalize Step ---
            normalized_states = {}
            if has_central_state:
                # Separate the float tensors from other types (like the boolean mask).
                central_states = current_states_dict["central"]
                central_float_states = {k: v for k, v in central_states.items() if torch.is_floating_point(v)}
                central_other_states = {k: v for k, v in central_states.items() if not torch.is_floating_point(v)}

                # Normalize ONLY the float tensors.
                normalized_central_floats = self.state_normalizer.normalize(central_float_states)

                # Recombine the normalized floats with the other tensors (the mask).
                normalized_states["central"] = {**normalized_central_floats, **central_other_states}

            if has_agent_state:
                normalized_states["agent"] = self.state_normalizer.normalize(
                    {"agent": current_states_dict["agent"]}
                )["agent"]
            
            states_for_agent_action = normalized_states

        states_for_agent_action_batched: Dict[str, Any] = {}
        if has_central_state: # Check if 'central' key exists after potential normalization
            if "central" in states_for_agent_action and states_for_agent_action["central"]:
                 states_for_agent_action_batched["central"] = {
                    k: v.unsqueeze(0) for k, v in states_for_agent_action["central"].items()
                }
        if has_agent_state: # Check if 'agent' key exists after potential normalization
            if "agent" in states_for_agent_action and states_for_agent_action["agent"] is not None:
                 states_for_agent_action_batched["agent"] = states_for_agent_action["agent"].unsqueeze(0)

        # Ensure at least one state component is present for the agent
        if not states_for_agent_action_batched or \
           (not ("central" in states_for_agent_action_batched and states_for_agent_action_batched["central"]) and \
            not ("agent" in states_for_agent_action_batched and states_for_agent_action_batched["agent"] is not None)):
            print("RLRunner Error: No valid state data to pass to agent.get_actions().")
            # Decide how to handle - potentially send dummy actions or raise error
            # For now, similar to empty states_from_comm:
            if hasattr(self.agent, 'total_action_dim_per_agent') and self.agent.total_action_dim_per_agent > 0:
                num_total_actions_per_env = self.num_agents_cfg * self.agent.total_action_dim_per_agent
                dummy_actions = torch.zeros((self.num_envs, num_total_actions_per_env), device=self.device)
                self.agentComm.send_actions(dummy_actions)
            else:
                self.agentComm.send_actions(torch.empty(0, device=self.device))
            return


        actions_ue_flat, (log_probs, entropies, values, baselines), next_h = self.agent.get_actions(
            states_for_agent_action_batched,
            dones=dones_from_comm,
            truncs=truncs_from_comm,
            h_prev_batch=self.current_memory_hidden_states if self.enable_memory else None,
        )

        if self.enable_memory and next_h is not None:
            self.current_memory_hidden_states = next_h 

        self.agentComm.send_actions(actions_ue_flat)

        needs_mask = needs_action.squeeze(0).squeeze(-1)
        actions_shaped = actions_ue_flat.view(self.num_envs, self.num_agents_cfg, -1)
        for e in range(self.num_envs):
            if needs_mask[e] < 0.5:
                continue
            obs_e: Dict[str, Any] = {}
            if "central" in current_states_dict:
                obs_e["central"] = {k: v[e].clone() for k, v in current_states_dict["central"].items()}
            if "agent" in current_states_dict:
                obs_e["agent"] = current_states_dict["agent"][e].clone()

            # if there's a pending step from the previous call, assign next_obs and done/trunc
            if self.pending_steps[e]:
                prev_step = self.pending_steps[e][-1]
                prev_step["next_obs"] = obs_e
                prev_step["done"] = dones[e].unsqueeze(-1)
                prev_step["trunc"] = truncs[e].unsqueeze(-1)
            step_info = {
                "obs": obs_e,
                "action": actions_shaped[e].clone(),
                "log_prob": log_probs[e].clone(),
                "entropy": entropies[e].clone(),
                "value": values[e].clone(),
                "baseline": baselines[e].clone(),
            }
            self.pending_steps[e].append(step_info)
        
        # Pass the unbatched, potentially normalized state dict to StateRecorder
        if self.state_recorder:
            # Assuming StateRecorder will be updated to handle this dictionary:
            # current_states_dict is {"central": {"comp": tensor_EHMW}, "agent": tensor_E_NA_Obs}
            # StateRecorder might only care about specific components or the first env.
            # For now, passing the whole dict for the first environment for simplicity.
            # The StateRecorder needs adaptation to process this.
            first_env_states_for_recorder: Dict[str, Any] = {}
            if "central" in current_states_dict and current_states_dict["central"]:
                first_env_states_for_recorder["central"] = {
                    k: v[0].cpu().numpy() for k, v in current_states_dict["central"].items() # Data for env 0
                }
            if "agent" in current_states_dict and current_states_dict["agent"] is not None:
                first_env_states_for_recorder["agent"] = current_states_dict["agent"][0].cpu().numpy() # Data for env 0
            
            if first_env_states_for_recorder: # Only record if there's something to record
                 self.state_recorder.record_frame(first_env_states_for_recorder)


    def _handle_update(self):
        self.update_idx += 1
        print(f"RLRunner update {self.update_idx}")

        # retrieve experiences (states are ignored except for bootstrapping)
        states_tensor, next_states_tensor, _, rewards_tensor, dones_tensor, truncs_tensor = self.agentComm.get_experiences()
        
        if not rewards_tensor.numel(): # No experiences received
            print("RLRunner: no experiences received in update. Skipping.")
            # Still need to signal UE that Python is done with this "empty" update cycle.
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event)
            return

        B_ue, NumEnv, _ = rewards_tensor.shape

        # only need rewards/dones/truncs - permute for env-major indexing
        rewards_tensor_p = rewards_tensor.permute(1, 0, 2).contiguous()
        dones_tensor_p = dones_tensor.permute(1, 0, 2).contiguous()
        truncs_tensor_p = truncs_tensor.permute(1, 0, 2).contiguous()

        # --- Bootstrap value for final next states ---
        bootstrap_states: Dict[str, Any] = {}
        if next_states_tensor.get("central"):
            bootstrap_states["central"] = {
                k: v[B_ue - 1] for k, v in next_states_tensor["central"].items()
            }
        if next_states_tensor.get("agent") is not None:
            bootstrap_states["agent"] = next_states_tensor["agent"][B_ue - 1]

        if self.state_normalizer and bootstrap_states:
            normalized_bootstrap: Dict[str, Any] = {}
            if "central" in bootstrap_states:
                central = bootstrap_states["central"]
                central_float = {k: v for k, v in central.items() if torch.is_floating_point(v)}
                central_other = {k: v for k, v in central.items() if not torch.is_floating_point(v)}
                norm_c = self.state_normalizer.normalize(central_float)
                normalized_bootstrap["central"] = {**norm_c, **central_other}
            if "agent" in bootstrap_states:
                normalized_bootstrap["agent"] = self.state_normalizer.normalize({"agent": bootstrap_states["agent"]})["agent"]
            bootstrap_states = normalized_bootstrap

        bootstrap_states_batched: Dict[str, Any] = {}
        if "central" in bootstrap_states:
            bootstrap_states_batched["central"] = {k: v.unsqueeze(0) for k, v in bootstrap_states["central"].items()}
        if "agent" in bootstrap_states:
            bootstrap_states_batched["agent"] = bootstrap_states["agent"].unsqueeze(0)

        bootstrap_dones = dones_tensor[B_ue - 1].unsqueeze(0)
        bootstrap_truncs = truncs_tensor[B_ue - 1].unsqueeze(0)

        with torch.no_grad():
            if bootstrap_states_batched:
                if self.enable_memory and self.current_memory_hidden_states is not None:
                    reset_mask = (bootstrap_dones.squeeze(-1) > 0.5) | (
                        bootstrap_truncs.squeeze(-1) > 0.5
                    )
                    for env_i, do_reset in enumerate(reset_mask):
                        if do_reset.item():
                            self.current_memory_hidden_states[env_i].zero_()
                _, (_, _, bootstrap_values, _), _ = self.agent.get_actions(
                    bootstrap_states_batched,
                    dones=bootstrap_dones,
                    truncs=bootstrap_truncs,
                    eval=True,
                    h_prev_batch=self.current_memory_hidden_states if self.enable_memory else None,
                )
            else:
                bootstrap_values = torch.zeros(NumEnv, 1, device=self.device)

        for e in range(NumEnv):
            for t in range(B_ue):
                reward_step = rewards_tensor_p[e, t]
                step_info = self.pending_steps[e].pop(0) if self.pending_steps[e] else None
                if step_info is None:
                    continue
                obs_step = step_info["obs"]
                next_obs_step = step_info.get("next_obs", {})
                if not next_obs_step:
                    next_obs_step = {}
                    if next_states_tensor.get("central"):
                        next_obs_step["central"] = {
                            k: v[t, e].clone()
                            for k, v in next_states_tensor["central"].items()
                        }
                    if next_states_tensor.get("agent") is not None:
                        next_obs_step["agent"] = next_states_tensor["agent"][t, e].clone()
                done_step = step_info.get("done", dones_tensor_p[e, t])
                trunc_step = step_info.get("trunc", truncs_tensor_p[e, t])
                action_step = step_info["action"]
                logp_step = step_info["log_prob"]
                ent_step = step_info["entropy"]
                val_step = step_info["value"]
                base_step = step_info["baseline"]

                self.current_segments[e].add_step(
                    obs_step,
                    action_step,
                    reward_step,
                    next_obs_step,
                    done_step,
                    trunc_step,
                    logp_step,
                    val_step,
                    base_step,
                    ent_step,
                )

                term_now = (done_step.item() > 0.5) or (trunc_step.item() > 0.5)
                if self.current_segments[e].is_full() or term_now:
                    self.current_episode_segments[e].append(self.current_segments[e])
                    if term_now:
                        self._finalize_episode(e)
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    if self.enable_memory and self.current_memory_hidden_states is not None:
                        self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        if not self.pad_trajectories:
            for e in range(NumEnv):
                seg = self.current_segments[e]
                if seg.true_sequence_length > 0:
                    self.current_episode_segments[e].append(seg)
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    if self.enable_memory and self.current_memory_hidden_states is not None:
                         self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        # finalize any remaining segments using bootstrapped value
        for e in range(NumEnv):
            self._finalize_rollout(e, bootstrap_values[e])
        
        all_completed_segments = [seg for env_segments in self.completed_segments for seg in env_segments]
        for env_segments in self.completed_segments: 
            env_segments.clear()

        if not all_completed_segments:
            print("RLRunner: no completed segments to update with – skipping agent update.")
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event) # Corrected call
            return

        (
            batch_obs_update,
            batch_act_update,
            batch_rew_update,
            batch_nobs_update,
            batch_done_update,
            batch_trunc_update,
            batch_logp_update,
            batch_val_update,
            batch_base_update,
            batch_entropy_update,
            batch_returns_update,
            batch_init_h_update,
            batch_attn_mask_update,
        ) = self._collate_and_pad_sequences(all_completed_segments)

        if not batch_returns_update.numel():
            print("RLRunner: collation resulted in empty batch. Skipping agent update.")
            if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
                 win32event.SetEvent(self.agentComm.update_received_event) # Corrected call
            return
        
        if self.state_normalizer:
            # --- Update Step ---
            # (Update logic for batch_obs_update and batch_nobs_update remains as before)
            if "central" in batch_obs_update:
                central_obs_to_update = {k: v for k, v in batch_obs_update["central"].items() if torch.is_floating_point(v)}
                if central_obs_to_update: self.state_normalizer.update(central_obs_to_update)
            if "agent" in batch_obs_update and torch.is_floating_point(batch_obs_update["agent"]):
                self.state_normalizer.update({"agent": batch_obs_update["agent"]})
            if "central" in batch_nobs_update:
                central_nobs_to_update = {k: v for k, v in batch_nobs_update["central"].items() if torch.is_floating_point(v)}
                if central_nobs_to_update: self.state_normalizer.update(central_nobs_to_update)
            if "agent" in batch_nobs_update and torch.is_floating_point(batch_nobs_update["agent"]):
                self.state_normalizer.update({"agent": batch_nobs_update["agent"]})

            # --- Normalize Step ---
            # Normalize batch_obs_update
            central_obs = batch_obs_update.get("central", {})
            obs_float_states = {k: v for k, v in central_obs.items() if torch.is_floating_point(v)}
            obs_other_states = {k: v for k, v in central_obs.items() if not torch.is_floating_point(v)}
            normalized_obs_floats = self.state_normalizer.normalize(obs_float_states)
            
            normalized_batch_obs = {"central": {**normalized_obs_floats, **obs_other_states}}
            if "agent" in batch_obs_update:
                normalized_batch_obs["agent"] = self.state_normalizer.normalize({"agent": batch_obs_update["agent"]})["agent"]
            batch_obs_update = normalized_batch_obs

            # Normalize batch_nobs_update
            central_nobs = batch_nobs_update.get("central", {})
            nobs_float_states = {k: v for k, v in central_nobs.items() if torch.is_floating_point(v)}
            nobs_other_states = {k: v for k, v in central_nobs.items() if not torch.is_floating_point(v)}
            normalized_nobs_floats = self.state_normalizer.normalize(nobs_float_states)

            normalized_batch_nobs = {"central": {**normalized_nobs_floats, **nobs_other_states}}
            if "agent" in batch_nobs_update:
                normalized_batch_nobs["agent"] = self.state_normalizer.normalize({"agent": batch_nobs_update["agent"]})["agent"]
            batch_nobs_update = normalized_batch_nobs

        logs = self.agent.update(
            batch_obs_update,
            batch_act_update,
            batch_returns_update,
            batch_rew_update,
            batch_logp_update,
            batch_val_update,
            batch_base_update,
            batch_entropy_update,
            batch_nobs_update,
            batch_done_update,
            batch_trunc_update,
            batch_init_h_update,
            batch_attn_mask_update,
        )
        self._log_step(logs)

        if self.update_idx % self.save_freq == 0:
            self.agent.save(f"model_update_{self.update_idx}.pth") 
            print("Model checkpoint saved.")
        
        if hasattr(self.agentComm, 'update_received_event') and self.agentComm.update_received_event:
            win32event.SetEvent(self.agentComm.update_received_event)
        else:
            print("RLRunner Warning: agentComm.update_received_event not found or is None in _handle_update.")

    def _collate_and_pad_sequences(self, segments: List[TrajectorySegment]) -> Tuple[
        Dict[str, Any], torch.Tensor, torch.Tensor, Dict[str, Any],
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        Optional[torch.Tensor], torch.Tensor
    ]:
        if not segments:
            empty_dict: Dict[str, Any] = {}
            empty_tensor = torch.empty(0, device=self.device)
            return (
                empty_dict, empty_tensor, empty_tensor, empty_dict,
                empty_tensor, empty_tensor, empty_tensor, empty_tensor,
                empty_tensor, empty_tensor, empty_tensor, None, empty_tensor
            )

        target_seq_len = self.sequence_length if self.pad_trajectories else max(s.true_sequence_length for s in segments)
        
        batch_observations: Dict[str, Any] = {}
        batch_next_observations: Dict[str, Any] = {}
        
        # Check if "central" and "agent" keys are expected based on the first segment.
        # This assumes consistent structure across segments.
        first_obs_example = segments[0].observations[0] if segments[0].observations else {}
        expect_central = "central" in first_obs_example and isinstance(first_obs_example["central"], dict)
        expect_agent = "agent" in first_obs_example and first_obs_example["agent"] is not None

        central_component_keys: List[str] = []
        if expect_central:
            batch_observations["central"] = {}
            batch_next_observations["central"] = {}
            central_component_keys = list(first_obs_example["central"].keys())
            for key in central_component_keys:
                batch_observations["central"][key] = []
                batch_next_observations["central"][key] = []
        
        if expect_agent:
            batch_observations["agent"] = []
            batch_next_observations["agent"] = []
        
        A_list, R_list, D_list, TR_list = [], [], [], []
        LP_list, V_list, B_list, ENT_list, RET_list = [], [], [], [], []
        H_list, lens_list = [], []

        def _get_dummy_shape_and_dtype(obs_example_dict: Dict[str, Any], key_path: List[str]) -> Tuple[Tuple, torch.dtype]:
            # Helper to get shape and dtype from a potentially nested example observation
            current_level = obs_example_dict
            for k_part in key_path[:-1]: # Navigate to parent dict
                current_level = current_level.get(k_part, {})
            
            example_tensor = current_level.get(key_path[-1])
            if example_tensor is not None and isinstance(example_tensor, torch.Tensor):
                return example_tensor.shape, example_tensor.dtype
            
            # Fallback if specific component is missing in the first example (should ideally not happen if structure is consistent)
            # This fallback is very basic and might need more sophisticated handling based on expected types/shapes
            if key_path[0] == "agent": return (self.num_agents_cfg, 1), torch.float32 # Guess agent obs dim as 1
            if key_path[0] == "central" and len(key_path) > 1: # e.g. central_some_component
                # Try to find shape from config if possible or use a default
                return (1,1), torch.float32 # Guess H,W as 1,1 for a central component
            return tuple(), torch.float32


        def _pad_sequence_of_tensors(tensor_list: List[torch.Tensor], pad_val: float = 0.0, 
                                     ref_obs_for_shape: Optional[Dict[str, Any]] = None,
                                     ref_key_path: Optional[List[str]] = None) -> torch.Tensor:
            if not tensor_list:
                if target_seq_len > 0 and ref_obs_for_shape and ref_key_path:
                    dummy_shape, dummy_dtype = _get_dummy_shape_and_dtype(ref_obs_for_shape, ref_key_path)
                    if dummy_shape: # Ensure dummy_shape is not empty
                         return torch.full((target_seq_len, *dummy_shape), pad_val, device=self.device, dtype=dummy_dtype)
                print(f"Warning: _pad_sequence_of_tensors received an empty list and could not determine dummy shape for key path {ref_key_path}.")
                return torch.empty((target_seq_len, 0), device=self.device, dtype=torch.float32) # Fallback to a possibly problematic empty tensor

            current_sequence = torch.stack(tensor_list, dim=0) 
            actual_seq_len = current_sequence.shape[0]

            if actual_seq_len == target_seq_len: return current_sequence
            elif actual_seq_len > target_seq_len: return current_sequence[:target_seq_len]
            else:
                pad_len = target_seq_len - actual_seq_len
                padding_shape = (pad_len, *current_sequence.shape[1:])
                pad_tensor = torch.full(padding_shape, pad_val, device=self.device, dtype=current_sequence.dtype)
                return torch.cat([current_sequence, pad_tensor], dim=0)

        first_segment_first_obs_example = segments[0].observations[0] if segments and segments[0].observations else {}

        for seg in segments:
            obs_steps, act_steps, rew_steps, next_obs_steps, done_steps, trunc_steps, logp_steps, val_steps, base_steps, ent_steps, ret_steps, h0_step, len_step = seg.tensors_for_collation()
            lens_list.append(min(len_step, target_seq_len))

            if expect_agent:
                agent_obs_sequence = [s.get("agent") for s in obs_steps if s.get("agent") is not None]
                batch_observations["agent"].append(_pad_sequence_of_tensors(agent_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["agent"]))
                
                agent_next_obs_sequence = [ns.get("agent") for ns in next_obs_steps if ns.get("agent") is not None]
                batch_next_observations["agent"].append(_pad_sequence_of_tensors(agent_next_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["agent"]))

            if expect_central:
                for comp_key in central_component_keys:
                    central_comp_obs_sequence = [s.get("central", {}).get(comp_key) for s in obs_steps if s.get("central", {}).get(comp_key) is not None]
                    batch_observations["central"][comp_key].append(_pad_sequence_of_tensors(central_comp_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["central", comp_key]))

                    central_comp_next_obs_sequence = [ns.get("central", {}).get(comp_key) for ns in next_obs_steps if ns.get("central", {}).get(comp_key) is not None]
                    batch_next_observations["central"][comp_key].append(_pad_sequence_of_tensors(central_comp_next_obs_sequence, ref_obs_for_shape=first_segment_first_obs_example, ref_key_path=["central", comp_key]))
            
            A_list.append(_pad_sequence_of_tensors(act_steps, ref_obs_for_shape=None))
            R_list.append(_pad_sequence_of_tensors(rew_steps, ref_obs_for_shape=None))
            D_list.append(_pad_sequence_of_tensors(done_steps, pad_val=1.0, ref_obs_for_shape=None))
            TR_list.append(_pad_sequence_of_tensors(trunc_steps, pad_val=1.0, ref_obs_for_shape=None))
            LP_list.append(_pad_sequence_of_tensors(logp_steps, ref_obs_for_shape=None))
            V_list.append(_pad_sequence_of_tensors(val_steps, ref_obs_for_shape=None))
            B_list.append(_pad_sequence_of_tensors(base_steps, ref_obs_for_shape=None))
            ENT_list.append(_pad_sequence_of_tensors(ent_steps, ref_obs_for_shape=None))
            RET_list.append(_pad_sequence_of_tensors(ret_steps, ref_obs_for_shape=None))
            
            if self.enable_memory and h0_step is not None: H_list.append(h0_step)

        final_batch_obs: Dict[str, Any] = {}
        if expect_agent and batch_observations["agent"] and any(t.numel() > 0 for t in batch_observations["agent"]):
            final_batch_obs["agent"] = torch.stack([t for t in batch_observations["agent"] if t.numel() > 0], dim=0)
        
        if expect_central:
            final_batch_obs["central"] = {}
            for comp_key in central_component_keys:
                if batch_observations["central"][comp_key] and any(t.numel() > 0 for t in batch_observations["central"][comp_key]):
                    final_batch_obs["central"][comp_key] = torch.stack([t for t in batch_observations["central"][comp_key] if t.numel() > 0], dim=0)
        
        final_batch_nobs: Dict[str, Any] = {}
        if expect_agent and batch_next_observations["agent"] and any(t.numel() > 0 for t in batch_next_observations["agent"]):
            final_batch_nobs["agent"] = torch.stack([t for t in batch_next_observations["agent"] if t.numel() > 0], dim=0)
        
        if expect_central:
            final_batch_nobs["central"] = {}
            for comp_key in central_component_keys:
                if batch_next_observations["central"][comp_key] and any(t.numel() > 0 for t in batch_next_observations["central"][comp_key]):
                    final_batch_nobs["central"][comp_key] = torch.stack([t for t in batch_next_observations["central"][comp_key] if t.numel() > 0], dim=0)
        
        def stack_if_not_empty(tensor_list):
            filtered_list = [t for t in tensor_list if t.numel() > 0]
            return torch.stack(filtered_list, dim=0) if filtered_list else torch.empty(0, device=self.device)

        batch_actions = stack_if_not_empty(A_list)
        batch_rewards = stack_if_not_empty(R_list)
        batch_dones = stack_if_not_empty(D_list)
        batch_truncs = stack_if_not_empty(TR_list)
        batch_logp = stack_if_not_empty(LP_list)
        batch_values = stack_if_not_empty(V_list)
        batch_baselines = stack_if_not_empty(B_list)
        batch_entropies = stack_if_not_empty(ENT_list)
        batch_returns = stack_if_not_empty(RET_list)
        
        batch_initial_h = torch.stack(H_list, dim=0) if (self.enable_memory and H_list) else None
        
        num_segments_in_batch = len(lens_list)
        attention_mask = torch.zeros(num_segments_in_batch, target_seq_len, device=self.device)
        for i, l_eff in enumerate(lens_list):
            attention_mask[i, :l_eff] = 1.0
            
        return (
            final_batch_obs,
            batch_actions,
            batch_rewards,
            final_batch_nobs,
            batch_dones,
            batch_truncs,
            batch_logp,
            batch_values,
            batch_baselines,
            batch_entropies,
            batch_returns,
            batch_initial_h,
            attention_mask,
        )

    def _log_step(self, logs: Dict[str, Any]):
        for k, v_val in logs.items():
            if isinstance(v_val, torch.Tensor):
                scalar_v = v_val.item() if v_val.numel() == 1 else v_val.mean().item() 
            elif isinstance(v_val, (list, np.ndarray)):
                 scalar_v = np.mean(v_val).item() if len(v_val) > 0 else 0.0 
            elif isinstance(v_val, (int, float)):
                scalar_v = float(v_val)
            else:
                try: scalar_v = float(v_val)
                except (TypeError, ValueError):
                    print(f"RLRunner Log: Skipping log for key '{k}' due to unconvertible type: {type(v_val)}")
                    continue
            self.writer.add_scalar(k, scalar_v, self.update_idx)

    def end(self):
        if hasattr(self, 'agentComm') and self.agentComm:
            self.agentComm.cleanup()
        if hasattr(self, 'writer') and self.writer:
            self.writer.close()
        # Save StateRecorder video if it exists and has frames
        if self.state_recorder and hasattr(self.state_recorder, 'frames') and len(self.state_recorder.frames) > 0:
            print("RLRunner ending: Saving any remaining StateRecorder frames...")
            self.state_recorder.save_video()
            self.state_recorder.frames.clear() # Clear frames after saving
        print("RLRunner ended.")
