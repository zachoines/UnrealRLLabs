from Config import MAPOCAConfig, TrainInfo, ObservationSpace, ActionSpace
from Agents import Agent
from Agents.MA_POCA import MAPocaAgent, MAPOCAConfig
from Agents.A2C import A2C, A2CConfig
from Environment import SharedMemoryInterface
from Runner import RLRunner

if __name__ == "__main__":
    memory_interface = SharedMemoryInterface()
    config = memory_interface.config
    train_info = TrainInfo(**config["TrainInfo"])
    obs_space = ObservationSpace(config['EnvironmentInfo']['StateSize'], **config['EnvironmentInfo']['MultiAgent'])
    action_space = ActionSpace(**config['EnvironmentInfo']['ActionSpace'])

    agent = MAPocaAgent( # TODO: Extract Agent type from received config info
        MAPOCAConfig(
            obs_space,
            action_space
        )
    )
    runner = RLRunner(agent=agent, agentComm=memory_interface)
    
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()

# tensorboard --logdir runs --host localhost --port 8888