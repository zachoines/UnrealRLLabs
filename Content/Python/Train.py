import argparse
import json
from Source.Factory import AgentEnvFactory
import os

def main():
    parser = argparse.ArgumentParser(description='Train MA-POCA Agent')
    parser.add_argument('--config', type=str, default='Configs/TerraShift.json',
                        help='Path to JSON config file.')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Override checkpoint path specified in the config file.')
    args = parser.parse_args()
    # Load config from JSON
    with open(args.config, 'r', encoding='utf-8-sig') as f:
        config = json.load(f)

    # Create agent, environment interface, etc.
    factory = AgentEnvFactory(config)
    agent, agentComm = factory.create_agent_and_environment()

    train_cfg = config.get("train", {})
    checkpoint_cfg = train_cfg.get("checkpoint", {}) if isinstance(train_cfg, dict) else {}
    cfg_checkpoint_path = checkpoint_cfg.get("path")
    restore_optimizers = checkpoint_cfg.get("restore_optimizers", True)
    restore_schedulers = checkpoint_cfg.get("restore_schedulers", True)

    resume_path = args.resume_from_checkpoint or cfg_checkpoint_path
    if resume_path:
        expanded_path = os.path.expandvars(os.path.expanduser(resume_path))
        if os.path.exists(expanded_path):
            print(f"Resuming training from checkpoint: {expanded_path}")
            if not restore_optimizers:
                print("  - Optimizer state will NOT be restored (per configuration).")
            if not restore_schedulers:
                print("  - Scheduler state will NOT be restored and will be reset to defaults (per configuration).")
            agent.load(
                expanded_path,
                load_optimizers=restore_optimizers,
                load_schedulers=restore_schedulers,
                reset_schedulers=not restore_schedulers
            )
        else:
            print(f"Warning: Checkpoint file not found at {expanded_path}. Starting from scratch.")

    runner = factory.create_runner(agent, agentComm)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()

if __name__ == "__main__":
    main()

# tensorboard --logdir runs --host localhost --port 8888
