# NOTICE: This file includes modifications generated with the assistance of generative AI (VSCode Copilot Assistant).
# Original code structure and logic by the project author.
# The modifications are intended to enhance the functionality and performance of the code.
# The author has reviewed all changes for correctness.
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from Source.Agent import Agent
from Source.Utility import RunningMeanStdNormalizer
from Source.Environment import EnvCommunicationInterface, EventType

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
        self.observations: List[Dict[str, torch.Tensor]] = []
        self.next_observations: List[Dict[str, torch.Tensor]] = []
        self.actions: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.truncs: List[torch.Tensor] = []
        self.initial_hidden_state: Optional[torch.Tensor] = None

    def is_full(self) -> bool:
        return self.enable_padding and self.true_sequence_length >= self.max_segment_length

    def add_step(self, obs, act, rew, next_obs, done, trunc):
        if self.is_full():
            print("RLRunner warn: add_step called on full segment")
            return
        self.observations.append(obs)
        self.next_observations.append(next_obs)
        self.actions.append(act)
        self.rewards.append(rew)
        self.dones.append(done)
        self.truncs.append(trunc)
        self.true_sequence_length += 1

    def tensors_for_collation(self):
        return (
            self.observations,
            self.actions,
            self.rewards,
            self.next_observations,
            self.dones,
            self.truncs,
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
        if "agent" in env_shape["state"]:
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
        if self.enable_memory:
            for e in range(self.num_envs):
                self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()
        self.completed_segments: List[List[TrajectorySegment]] = [[] for _ in range(self.num_envs)]

        norm_cfg = trn_cfg.get("states_normalizer", None)
        self.state_normalizer = RunningMeanStdNormalizer(**norm_cfg, device=self.device) if norm_cfg else None

        self.writer = SummaryWriter()
        self.update_idx = 0
        print("RLRunner initialised (envs:", self.num_envs, ")")

    # =================================================================
    def start(self):
        while True:
            evt = self.agentComm.wait_for_event()
            if evt == EventType.GET_ACTIONS:
                self._handle_get_actions()
            elif evt == EventType.UPDATE:
                self._handle_update()

    # -----------------------------------------------------------------
    def _handle_get_actions(self):
        states, dones, truncs = self.agentComm.get_states()
        dones = dones.squeeze(0).squeeze(-1)  # (E,)
        truncs = truncs.squeeze(0).squeeze(-1)

        if self.enable_memory:
            for e in range(self.num_envs):
                if (dones[e] > 0.5) or (truncs[e] > 0.5):
                    self.current_memory_hidden_states[e].zero_()
                    if self.current_segments[e].true_sequence_length > 0:
                        self.completed_segments[e].append(self.current_segments[e])
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        states_for_agent = states
        if self.state_normalizer:
            self.state_normalizer.update(states)
            states_for_agent = self.state_normalizer.normalize(states)

        actions, _, next_h = self.agent.get_actions(
            states_for_agent,
            dones=dones,
            truncs=truncs,
            h_prev_batch=self.current_memory_hidden_states if self.enable_memory else None,
        )
        if self.enable_memory:
            self.current_memory_hidden_states = next_h
        self.agentComm.send_actions(actions)

    # -----------------------------------------------------------------
    def _handle_update(self):
        self.update_idx += 1
        print(f"RLRunner update {self.update_idx}")

        s, s_next, a, r, d, tr = self.agentComm.get_experiences()
        steps_recv = r.shape[0]

        for e in range(self.num_envs):
            for t in range(steps_recv):
                obs = {k: v[t, e] for k, v in s.items() if v is not None}
                nxt = {k: v[t, e] for k, v in s_next.items() if v is not None}
                self.current_segments[e].add_step(obs, a[t, e], r[t, e], nxt, d[t, e], tr[t, e])
                term_now = (d[t, e] > 0.5) or (tr[t, e] > 0.5)
                if self.current_segments[e].is_full() or term_now:
                    self.completed_segments[e].append(self.current_segments[e])
                    self.current_segments[e] = TrajectorySegment(
                        self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                    )
                    if self.enable_memory:
                        self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        for e in range(self.num_envs):
            seg = self.current_segments[e]
            if seg.true_sequence_length > 0 and (not self.pad_trajectories):
                self.completed_segments[e].append(seg)
                self.current_segments[e] = TrajectorySegment(
                    self.num_agents_cfg, self.device, self.pad_trajectories, self.sequence_length
                )
                if self.enable_memory:
                    self.current_segments[e].initial_hidden_state = self.current_memory_hidden_states[e].clone()

        segments = [s for buf in self.completed_segments for s in buf]
        for buf in self.completed_segments:
            buf.clear()
        if not segments:
            print("RLRunner: no segments to update – skipping")
            return

        batch = self._collate_and_pad_sequences(segments)
        logs = self.agent.update(*batch)
        self._log_step(logs)
        if self.update_idx % self.save_freq == 0:
            torch.save(self.agent.state_dict(), f"model_update_{self.update_idx}.pth")
            print("Model checkpoint saved")

    # -----------------------------------------------------------------
    def _collate_and_pad_sequences(self, segments: List[TrajectorySegment]):
        if not segments:
            empty = {}
            z = torch.empty(0, device=self.device)
            return empty, z, z, empty, z, z, None, z

        tgt = self.sequence_length if self.pad_trajectories else max(s.true_sequence_length for s in segments)
        obs_keys = segments[0].observations[0].keys()

        def pad(x_list: List[torch.Tensor], pad_val=0.0):
            t = torch.stack(x_list)
            if t.shape[0] == tgt:
                return t
            if t.shape[0] > tgt:
                return t[: tgt]
            pad_len = tgt - t.shape[0]
            pad_tensor = torch.full((pad_len, *t.shape[1:]), pad_val, device=self.device, dtype=t.dtype)
            return torch.cat([t, pad_tensor], 0)

        batch_obs, batch_nobs = {k: [] for k in obs_keys}, {k: [] for k in obs_keys}
        A, R, D, TR, H, lens = [], [], [], [], [], []
        for seg in segments:
            o, a, r, n, d, t, h0, L = seg.tensors_for_collation()
            lens.append(L)
            for k in obs_keys:
                batch_obs[k].append(pad([x[k] for x in o]))
                batch_nobs[k].append(pad([x[k] for x in n]))
            A.append(pad(a))
            R.append(pad(r))
            D.append(pad(d, 1.0))
            TR.append(pad(t, 1.0))
            if self.enable_memory and h0 is not None:
                H.append(h0)

        B = len(lens)
        action_dim = A[0].shape[-1] if A and A[0] is not None else self.agent.total_action_dim_per_agent
        act_tensor = (
            torch.stack(A, 0)
            if A and A[0] is not None
            else torch.zeros(B, tgt, self.num_agents_cfg, action_dim, device=self.device)
        )
        batch_obs = {k: torch.stack(v, 0) for k, v in batch_obs.items()}
        batch_nobs = {k: torch.stack(v, 0) for k, v in batch_nobs.items()}
        batch_rew, batch_done, batch_trunc = map(lambda lst: torch.stack(lst, 0), (R, D, TR))
        init_h = torch.stack(H, 0) if (self.enable_memory and H) else None
        attn = torch.zeros(B, tgt, device=self.device)
        for i, l in enumerate(lens):
            attn[i, : min(l, tgt)] = 1.0
        return batch_obs, act_tensor, batch_rew, batch_nobs, batch_done, batch_trunc, init_h, attn

    # -----------------------------------------------------------------
    def _log_step(self, logs: Dict[str, Any]):
        for k, v in logs.items():
            self.writer.add_scalar(k, float(v), self.update_idx)

    def end(self):
        self.agentComm.cleanup()
        self.writer.close()
