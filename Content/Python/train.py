import win32event
import win32api
import numpy as np
import time
from tqdm import tqdm
from multiprocessing import shared_memory

MUTEX_ALL_ACCESS = 0x1F0001
SLEEP_INTERVAL = 1.0  # seconds

class SharedMemoryAgentProcessorConfig:
    def __init__(self, num_environments: int, num_actions: int, state_size: int, training_batch_size: int):
        self.num_environments = num_environments
        self.num_actions = num_actions
        self.state_size = state_size
        self.training_batch_size = training_batch_size

class SharedMemoryAgentProcessor:
    def __init__(self, config: SharedMemoryAgentProcessorConfig):
        self.config = config

        # Initialize shared memories for actions, states, and updates
        self.action_shared_memory = self.wait_for_resource(shared_memory.SharedMemory, name="ActionsSharedMemory")
        self.states_shared_memory = self.wait_for_resource(shared_memory.SharedMemory, name="StatesSharedMemory")
        self.update_shared_memory = self.wait_for_resource(shared_memory.SharedMemory, name="UpdateSharedMemory")

        # Initialize synchronization events
        self.action_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")

        # Initialize synchronization mutexes
        self.actions_mutex = self.wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ActionsDataMutex")
        self.states_mutex = self.wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "StatesDataMutex")
        self.update_mutex = self.wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "UpdateDataMutex")

        self.actions_count = 0

    def wait_for_resource(self, func, *args, **kwargs):
        """Utility function to wait and retry until the resource is available."""
        dot_count = 1
        pbar = tqdm(total=4, desc="Waiting for Unreal Engine", position=0, leave=True, bar_format='{desc}')
        
        while True:
            try:
                resource = func(*args, **kwargs)
                if resource:
                    pbar.close()
                    return resource
            except Exception as _:
                pbar.set_description(f"Waiting for Unreal Engine {'.' * dot_count}")
                dot_count = (dot_count % 4) + 1  # Cycle between 1 and 4 dots
                pbar.update(dot_count - pbar.n)
                time.sleep(SLEEP_INTERVAL)

    def process(self):
        while True:
            result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)
            
            # Check for ActionReadyEvent
            if result == win32event.WAIT_OBJECT_0:
                win32event.WaitForSingleObject(self.states_mutex, win32event.INFINITE)  # Acquire the mutex

                try:
                    # Read states from shared memory
                    state_array = np.ndarray(shape=(self.config.num_environments, self.config.state_size), dtype=np.float32, buffer=self.states_shared_memory.buf)
                    states = state_array.copy()

                    # TODO: Process the states to compute corresponding actions.
                    # For now, we generate random actions as a placeholder.
                    actions = np.arange(start=self.actions_count, stop=self.actions_count + (self.config.num_environments * self.config.num_actions)).reshape((self.config.num_environments, self.config.num_actions))
                    self.actions_count += (self.config.num_environments * self.config.num_actions)
                    # Write these actions back to shared memory
                    action_array = np.ndarray(shape=(self.config.num_environments, self.config.num_actions), dtype=np.float32, buffer=self.action_shared_memory.buf)
                    np.copyto(action_array, actions)
                    
                except Exception as e:
                    print(f"Error during action computation: {e}")

                win32event.ReleaseMutex(self.states_mutex)  # Release the mutex
                win32event.SetEvent(self.action_received_event)

            # Check for UpdateReadyEvent
            elif result == win32event.WAIT_OBJECT_0 + 1:
                with tqdm(total=100, desc="Processing Update", position=0, leave=True) as pbar:  # Placeholder progress bar
                    for _ in range(100):
                        time.sleep(0.01)  # Sleep for a bit to simulate some work
                        pbar.update(1)

                win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)  # Acquire the mutex
                
                # For simplicity, just read and discard the update data for now.
                # TODO: Implement proper handling.
                _ = self.update_shared_memory.buf.tobytes()  # Discard the data

                win32event.ReleaseMutex(self.update_mutex)  # Release the mutex
                win32event.SetEvent(self.update_received_event)

    def cleanup(self):
        self.action_shared_memory.close()
        self.states_shared_memory.close()
        self.update_shared_memory.close()

        # Clean up the synchronization events and mutexes
        for handle in [self.action_ready_event, self.action_received_event, self.update_ready_event, self.update_received_event, self.actions_mutex, self.states_mutex, self.update_mutex]:
            win32api.CloseHandle(handle)

if __name__ == "__main__":
    agent_processor = SharedMemoryAgentProcessor(
        SharedMemoryAgentProcessorConfig(num_environments=128, num_actions=2, state_size=6, training_batch_size=32)
    )
    try:
        agent_processor.process()
    finally:
        agent_processor.cleanup()
