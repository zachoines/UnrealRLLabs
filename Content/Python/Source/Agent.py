import copy
import torch
import torch.nn as nn
from typing import Dict, Tuple

class Agent(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super(Agent, self).__init__()
        self.config = config
        self.device = device
        self.optimizers = {}
        self._initial_scheduler_states = {}

    def save(self, location: str, include_optimizers: bool = False) -> None:
        """Save model parameters and optionally optimizer + scheduler states.

        Backward-compatible: older checkpoints without 'schedulers' will still load.
        """
        if include_optimizers:
            checkpoint = {
                "model": self.state_dict(),
                "optimizers": {name: opt.state_dict() for name, opt in self.optimizers.items()},
            }
            # Save schedulers if present
            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
                sched_state = {}
                try:
                    import torch as _torch
                    from torch.optim.lr_scheduler import _LRScheduler as _LS
                except Exception:  # pragma: no cover
                    _LS = tuple()  # type: ignore
                for name, sched in self.schedulers.items():
                    if hasattr(sched, "state_dict") and callable(getattr(sched, "state_dict")):
                        try:
                            # Works for torch schedulers and our LinearValueScheduler
                            sched_state[name] = sched.state_dict()
                        except Exception:
                            pass
                if sched_state:
                    checkpoint["schedulers"] = sched_state

            # Store meta inferred from filename if available (e.g., update number)
            import re
            m = re.search(r"update_(\d+)", str(location))
            if m:
                checkpoint.setdefault("meta", {})["update_num"] = int(m.group(1))
            torch.save(checkpoint, location)
        else:
            torch.save(self.state_dict(), location)

    def load(
        self,
        location: str,
        load_optimizers: bool = False,
        load_schedulers: bool = True,
        reset_schedulers: bool = False,
    ) -> None:
        """Load model parameters and optionally optimizer + scheduler states.

        If no scheduler states are present, heuristically initialize scheduler steps
        from the update number embedded in the filename (pattern 'update_####').
        """
        state = torch.load(location, map_location=self.device)
        inferred_steps = None
        checkpoint_update_meta = None
        scheduler_msgs = []
        load_schedulers = bool(load_schedulers)
        reset_requested = reset_schedulers or (not load_schedulers)

        if isinstance(state, dict) and "model" in state:
            self.load_state_dict(state["model"])
            if load_optimizers and "optimizers" in state:
                for name, opt_state in state["optimizers"].items():
                    if name in self.optimizers:
                        self.optimizers[name].load_state_dict(opt_state)

            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
                sched_state = state.get("schedulers", None) if load_schedulers else None
                if load_schedulers and isinstance(sched_state, dict):
                    for name, sstate in sched_state.items():
                        if name in self.schedulers:
                            sched = self.schedulers[name]
                            if hasattr(sched, "load_state_dict") and callable(getattr(sched, "load_state_dict")):
                                try:
                                    sched.load_state_dict(sstate)
                                except Exception:
                                    pass
                    scheduler_msgs.append("loaded state from checkpoint")
                elif load_schedulers:
                    import re
                    m = re.search(r"update_(\d+)", str(location))
                    if m:
                        inferred_steps = int(m.group(1))
                        for sched in self.schedulers.values():
                            try:
                                if hasattr(sched, "last_epoch"):
                                    sched.last_epoch = inferred_steps
                                if hasattr(sched, "counter") and hasattr(sched, "total_iters"):
                                    sched.counter = min(inferred_steps, getattr(sched, "total_iters", inferred_steps))
                            except Exception:
                                continue
                        scheduler_msgs.append(f"inferred steps from filename => {inferred_steps}")
                    else:
                        scheduler_msgs.append("no scheduler state found; reset to init")
                else:
                    scheduler_msgs.append("skipped per configuration")
                checkpoint_update_meta = (state.get("meta", {}) or {}).get("update_num")
        else:
            self.load_state_dict(state)

        if reset_requested and hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
            self._reset_schedulers()
            scheduler_msgs.append("reset to defaults")

        try:
            print("[Agent.load] Restored checkpoint:")
            print(f"  file: {location}")
            if checkpoint_update_meta is not None:
                print(f"  meta.update_num (saved): {checkpoint_update_meta}")
            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict) and self.schedulers:
                if scheduler_msgs:
                    print(f"  schedulers: {'; '.join(scheduler_msgs)}")
                else:
                    print("  schedulers: no scheduler updates applied")
                for name, sched in self.schedulers.items():
                    try:
                        if hasattr(sched, "counter") and hasattr(sched, "total_iters"):
                            cv = sched.current_value() if hasattr(sched, "current_value") else None
                            print(f"    - {name}: counter={sched.counter}/{sched.total_iters} value={cv}")
                        elif hasattr(sched, "last_epoch"):
                            lr_str = None
                            if hasattr(sched, "get_last_lr"):
                                try:
                                    lr_str = ",".join([f"{x:.6g}" for x in sched.get_last_lr()])
                                except Exception:
                                    lr_str = None
                            print(f"    - {name}: last_epoch={sched.last_epoch} last_lr={lr_str}")
                    except Exception:
                        continue
            if hasattr(self, "optimizers") and isinstance(self.optimizers, dict) and self.optimizers:
                print("  optimizers: current LRs")
                for oname, opt in self.optimizers.items():
                    try:
                        lrs = [pg.get("lr", None) for pg in opt.param_groups]
                        lrs_fmt = ",".join([f"{float(lr):.6g}" for lr in lrs if lr is not None]) if lrs else ""
                        print(f"    - {oname}: {lrs_fmt}")
                    except Exception:
                        continue
        except Exception:
            pass

    def _reset_schedulers(self) -> None:
        if not hasattr(self, "schedulers") or not isinstance(self.schedulers, dict):
            return
        initial_states = getattr(self, "_initial_scheduler_states", None)
        for name, sched in self.schedulers.items():
            if sched is None:
                continue
            state_loaded = False
            if initial_states and name in initial_states:
                snapshot = initial_states[name]
                if snapshot is not None and hasattr(sched, "load_state_dict"):
                    try:
                        sched.load_state_dict(copy.deepcopy(snapshot))
                        state_loaded = True
                    except Exception:
                        state_loaded = False
            if not state_loaded:
                if hasattr(sched, "last_epoch"):
                    sched.last_epoch = -1
                if hasattr(sched, "_step_count"):
                    sched._step_count = 0  # type: ignore[attr-defined]
                if hasattr(sched, "counter"):
                    sched.counter = 0  # type: ignore[attr-defined]
                if hasattr(sched, "t"):
                    sched.t = 0  # type: ignore[attr-defined]

    def get_actions(self, states: torch.Tensor, dones=None, truncs=None, eval: bool = False, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def update(self, states: torch.Tensor, next_states: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, truncs: torch.Tensor) -> Dict[str, torch.Tensor]:
        raise NotImplementedError
    
    def total_grad_norm(self, params):
        # compute the global L2 norm across all param grads
        total = 0.0
        for p in params:
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total += param_norm.item() ** 2
        return torch.tensor(total**0.5)
        
