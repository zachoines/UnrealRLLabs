from Config import MAPOCALSTMConfig, MAPOCAConfig, TrainInfo, ObservationSpace, ActionSpace
from Agents import Agent
from Agents.MA_POCA import *
from Agents.MA_POCA_LSTM import *
from Agents.MA_POCA_LSTM_V2 import *
from Agents.A2C import *
from Environment import SharedMemoryInterface
from Runner import RLRunner

if __name__ == "__main__":
    memory_interface = SharedMemoryInterface()
    config = memory_interface.config
    train_info = TrainInfo(**config["TrainInfo"])
    obs_space = ObservationSpace(config['EnvironmentInfo']['StateSize'], **config['EnvironmentInfo']['MultiAgent'])
    action_space = ActionSpace(**config['EnvironmentInfo']['ActionSpace'])
    agent = MAPocaLSTMAgentLight(
        MAPOCA_LSTM_Light_Config(
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