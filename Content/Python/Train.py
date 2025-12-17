import argparse
import json
import os
from pathlib import Path

from Source.Factory import AgentEnvFactory

def main():
    parser = argparse.ArgumentParser(description='Train MA-POCA Agent')
    parser.add_argument('--config', type=str, default='Configs/TerraShift.json',
                        help='Path to JSON config file.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='checkpoints/model_update_3240.pth',
                        help='Override checkpoint path specified in the config file.')
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    def resolve_path(path_str: str, must_exist: bool = False) -> Path:
        """
        Resolve a path, preferring the current working directory first,
        then falling back to the directory containing Train.py. This keeps
        relative paths in configs working regardless of where the script
        is launched from.
        """
        expanded = Path(os.path.expandvars(os.path.expanduser(path_str)))
        candidates = [expanded] if expanded.is_absolute() else [
            Path.cwd() / expanded,
            script_dir / expanded
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()

        resolved = candidates[-1].resolve()
        if must_exist:
            raise FileNotFoundError(f"Path '{path_str}' could not be found. Tried: {[str(c) for c in candidates]}")
        return resolved

    # Load config from JSON
    config_path = resolve_path(args.config, must_exist=True)
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    # Create agent, environment interface, etc.
    factory = AgentEnvFactory(config)
    agent, agentComm = factory.create_agent_and_environment()

    train_cfg = config.get("train", {})
    checkpoint_cfg = train_cfg.get("checkpoint", {}) if isinstance(train_cfg, dict) else {}
    cfg_checkpoint_path = checkpoint_cfg.get("path")
    restore_optimizers = checkpoint_cfg.get("restore_optimizers", True)
    restore_schedulers = checkpoint_cfg.get("restore_schedulers", True)

    resume_path_raw = args.resume_from_checkpoint or cfg_checkpoint_path
    checkpoint_extras = None
    resolved_checkpoint_path: Path = None  # type: ignore
    if resume_path_raw:
        resolved_checkpoint_path = resolve_path(resume_path_raw)
        checkpoint_cfg["resolved_path"] = str(resolved_checkpoint_path)
        checkpoint_cfg["resolved_dir"] = str(resolved_checkpoint_path.parent)
        if resolved_checkpoint_path.exists():
            print(f"Resuming training from checkpoint: {resolved_checkpoint_path}")
            if not restore_optimizers:
                print("  - Optimizer state will NOT be restored (per configuration).")
            if not restore_schedulers:
                print("  - Scheduler state will NOT be restored and will be reset to defaults (per configuration).")
            checkpoint_extras = agent.load(
                str(resolved_checkpoint_path),
                load_optimizers=restore_optimizers,
                load_schedulers=restore_schedulers,
                reset_schedulers=not restore_schedulers
            )
        else:
            print(f"Warning: Checkpoint file not found at {resolved_checkpoint_path}. Starting from scratch.")
    else:
        # Still expose a resolved directory so the runner can save checkpoints consistently.
        default_ckpt_dir = script_dir / "checkpoints"
        checkpoint_cfg.setdefault("resolved_dir", str(default_ckpt_dir.resolve()))

    runner = factory.create_runner(agent, agentComm)
    if checkpoint_extras:
        runner.restore_checkpoint_extras(checkpoint_extras)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()

if __name__ == "__main__":
    main()

# tensorboard --logdir runs --host localhost --port 8888
