import argparse
import yaml
import sys
from Source.Factory import AgentEnvFactory
from Source.Runner import RLRunner

def main():
    parser = argparse.ArgumentParser(description='Train MA-POCA Agent')
    parser.add_argument('--config', type=str, default='Configs/TerraShift.yaml',
                        help='Path to config file.')
    args = parser.parse_args()

    # Load config from YAML
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create agent, environment interface, etc.
    factory = AgentEnvFactory(config)
    agent, agentComm = factory.create_agent_and_environment()
    runner = factory.create_runner(agent, agentComm)

    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()

if __name__ == "__main__":
    main()