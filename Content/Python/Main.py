from Agents import A2C
from Environment import SharedMemoryInterface
from Runner import RLRunner

if __name__ == "__main__":
    memory_interface = SharedMemoryInterface()
    agent = A2C(memory_interface.config)
    runner = RLRunner(agent=agent, agentComm=memory_interface)
    
    try:
        runner.start()
    except KeyboardInterrupt:
        runner.end()


# tensorboard --logdir runs --host localhost --port 8888
