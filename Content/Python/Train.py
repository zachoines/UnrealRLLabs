import argparse
import json
from Source.Factory import AgentEnvFactory
import os

def main():
    parser = argparse.ArgumentParser(description='Train MA-POCA Agent')
    parser.add_argument('--config', type=str, default='Configs/TerraShift.json',
                        help='Path to JSON config file.')
    parser.add_argument('--resume_from_checkpoint', type=str, default="model_update_400.pth",
                        help='Path to a saved model checkpoint (.pth) to resume training from.')
    args = parser.parse_args()

    # Load config from JSON
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Create agent, environment interface, etc.
    factory = AgentEnvFactory(config)
    agent, agentComm = factory.create_agent_and_environment()

    if args.resume_from_checkpoint:
        if os.path.exists(args.resume_from_checkpoint):
            print(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            agent.load(args.resume_from_checkpoint)
        else:
            print(f"Warning: Checkpoint file not found at {args.resume_from_checkpoint}. Starting from scratch.")

    runner = factory.create_runner(agent, agentComm)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()

if __name__ == "__main__":
    main()

# tensorboard --logdir runs --host localhost --port 8888