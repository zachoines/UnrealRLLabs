import win32event
import win32api
import mmap
import numpy as np
import struct
import time
from tqdm import tqdm

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

        self.action_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")

        self.actions_mutex = self.wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "ActionsDataMutex")
        self.update_mutex = self.wait_for_resource(win32event.OpenMutex, MUTEX_ALL_ACCESS, False, "UpdateDataMutex")

        self.action_map_file = open("ActionsSharedMemory", "r+b")
        self.update_map_file = open("UpdatesSharedMemory", "r+b")

        self.action_mapped_memory = mmap.mmap(self.action_map_file.fileno(), self.config.num_environments * self.config.num_actions * 4)
        self.update_mapped_memory = mmap.mmap(self.update_map_file.fileno(), 0)

    def wait_for_resource(self, func, *args):
        """Utility function to wait and retry until the resource is available."""
        dot_count = 1
        pbar = tqdm(total=4, desc="Waiting for Unreal Engine", position=0, leave=True, bar_format='{desc}')
        
        while True:
            try:
                resource = func(*args)
                if resource:
                    pbar.close()
                    return resource
            except Exception as _:
                pbar.set_description(f"Waiting for Unreal Engine {'.' * dot_count}")
                dot_count = (dot_count % 4) + 1  # Cycle between 1 and 4 dots
                pbar.update(dot_count - pbar.n)
                time.sleep(SLEEP_INTERVAL)
        pbar.close()

    def process(self):
        while True:
            result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)
            
            if result == win32event.WAIT_OBJECT_0:  # ActionReadyEvent
                win32event.WaitForSingleObject(self.actions_mutex, win32event.INFINITE)  # Acquire the mutex

                try:
                    actions = np.random.rand(self.config.num_environments, self.config.num_actions)
                    self.action_mapped_memory.seek(0)
                    for action in actions:
                        self.action_mapped_memory.write(struct.pack(f'{len(action)}f', *action))
                except Exception as e:
                    print(f"Error during action computation: {e}")

                win32event.ReleaseMutex(self.actions_mutex)  # Release the mutex
                win32event.SetEvent(self.action_received_event)

            elif result == win32event.WAIT_OBJECT_0 + 1:  # UpdateReadyEvent
                with tqdm(total=100, desc="Processing Update", position=0, leave=True) as pbar:  # Placeholder progress bar
                    for _ in range(100):
                        time.sleep(0.01)  # Sleep for a bit to simulate some work
                        pbar.update(1)

                win32event.WaitForSingleObject(self.update_mutex, win32event.INFINITE)  # Acquire the mutex
                
                _ = self.update_mapped_memory.read()  # Discard the data

                win32event.ReleaseMutex(self.update_mutex)  # Release the mutex
                win32event.SetEvent(self.update_received_event)

    def cleanup(self):
        self.action_mapped_memory.close()
        self.update_mapped_memory.close()

        self.action_map_file.close()
        self.update_map_file.close()

        for handle in [self.action_ready_event, self.action_received_event, self.update_ready_event, self.update_received_event, self.actions_mutex, self.update_mutex]:
            win32api.CloseHandle(handle)

if __name__ == "__main__":
    agent_processor = SharedMemoryAgentProcessor(
        SharedMemoryAgentProcessorConfig(num_environments=5, num_actions=2, state_size=4, training_batch_size=10)
    )
    try:
        agent_processor.process()
    finally:
        agent_processor.cleanup()
