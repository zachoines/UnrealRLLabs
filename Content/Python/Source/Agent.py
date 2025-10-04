import torch
import torch.nn as nn
from typing import Dict, Tuple

class Agent(nn.Module):
    def __init__(self, config: dict, device: torch.device):
        super(Agent, self).__init__()
        self.config = config
        self.device = device
        self.optimizers = {}

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

    def load(self, location: str, load_optimizers: bool = False) -> None:
        """Load model parameters and optionally optimizer + scheduler states.

        If no scheduler states are present, heuristically initialize scheduler steps
        from the update number embedded in the filename (pattern 'update_####').
        """
        state = torch.load(location, map_location=self.device)
        inferred_steps = None
        checkpoint_had_schedulers = False
        checkpoint_update_meta = None
        if isinstance(state, dict) and "model" in state:
            self.load_state_dict(state["model"])
            if load_optimizers and "optimizers" in state:
                for name, opt_state in state["optimizers"].items():
                    if name in self.optimizers:
                        self.optimizers[name].load_state_dict(opt_state)

            # Restore schedulers when available
            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
                sched_state = state.get("schedulers", None)
                if isinstance(sched_state, dict):
                    checkpoint_had_schedulers = True
                    for name, sstate in sched_state.items():
                        if name in self.schedulers:
                            sched = self.schedulers[name]
                            if hasattr(sched, "load_state_dict") and callable(getattr(sched, "load_state_dict")):
                                try:
                                    sched.load_state_dict(sstate)
                                except Exception:
                                    pass
                else:
                    # Heuristic: derive step from filename if possible
                    import re
                    m = re.search(r"update_(\d+)", str(location))
                    if m:
                        inferred_steps = int(m.group(1))
                        # Clamp to each scheduler's horizon where applicable
                        from torch.optim.lr_scheduler import _LRScheduler as _LS
                        for sched in self.schedulers.values():
                            try:
                                # Pytorch LR schedulers
                                if hasattr(sched, "last_epoch"):
                                    sched.last_epoch = inferred_steps
                                # Our LinearValueScheduler
                                if hasattr(sched, "counter") and hasattr(sched, "total_iters"):
                                    sched.counter = min(inferred_steps, getattr(sched, "total_iters", inferred_steps))
                            except Exception:
                                continue
                    checkpoint_had_schedulers = False
                checkpoint_update_meta = (state.get("meta", {}) or {}).get("update_num")
        else:
            self.load_state_dict(state)

        # QoL: print a concise summary of what was restored/inferred
        try:
            print("[Agent.load] Restored checkpoint:")
            print(f"  file: {location}")
            if checkpoint_update_meta is not None:
                print(f"  meta.update_num (saved): {checkpoint_update_meta}")
            if hasattr(self, "schedulers") and isinstance(self.schedulers, dict):
                if checkpoint_had_schedulers:
                    print("  schedulers: loaded state from checkpoint")
                else:
                    if inferred_steps is not None:
                        print(f"  schedulers: inferred steps from filename => {inferred_steps}")
                    else:
                        print("  schedulers: no state found; no inferred steps (reset to init)")
                # Print per-scheduler status
                for name, sched in self.schedulers.items():
                    try:
                        if hasattr(sched, "counter") and hasattr(sched, "total_iters"):
                            # LinearValueScheduler-like
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
            # Print per-optimizer LR snapshot
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
        
