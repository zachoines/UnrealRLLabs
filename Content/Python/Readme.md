import win32event
import win32api
import mmap
import numpy as np
import struct
import time

class SharedMemoryAgentProcessorConfig:
    def __init__(self, num_environments: int, num_actions: int, state_size: int, training_batch_size: int):
        self.num_environments = num_environments
        self.num_actions = num_actions
        self.state_size = state_size
        self.training_batch_size = training_batch_size

class SharedMemoryAgentProcessor:
    def __init__(self, config: SharedMemoryAgentProcessorConfig):
        self.config = config

        # Using the wait_for_resource method to acquire the resources
        self.action_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReadyEvent")
        self.action_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "ActionReceivedEvent")
        self.update_ready_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReadyEvent")
        self.update_received_event = self.wait_for_resource(win32event.OpenEvent, win32event.EVENT_ALL_ACCESS, False, "UpdateReceivedEvent")
        self.actions_memory = self.wait_for_resource(win32api.OpenFileMapping, win32api.FILE_MAP_ALL_ACCESS, False, "ActionsSharedMemory")
        self.update_memory = self.wait_for_resource(win32api.OpenFileMapping, win32api.FILE_MAP_ALL_ACCESS, False, "UpdateSharedMemory")

    def wait_for_resource(self, func, *args):
        """Utility function to wait and retry until the resource is available."""
        while True:
            try:
                resource = func(*args)
                if resource:
                    return resource
            except Exception as e:
                time.sleep(0.1)  # Sleep for 100 milliseconds before retrying

    def process(self):
        while True:
            # Wait for either ActionReadyEvent or UpdateReadyEvent
            result = win32event.WaitForMultipleObjects([self.action_ready_event, self.update_ready_event], False, win32event.INFINITE)
            
            if result == win32event.WAIT_OBJECT_0:  # ActionReadyEvent
                with mmap.mmap(self.actions_memory.handle, 0) as view:  # Auto-closes the mapped view
                    actions = np.random.rand(self.config.num_environments, self.config.num_actions)
                    view.seek(0)
                    for action in actions:
                        view.write(struct.pack(f'{len(action)}f', *action))

                # Signal that the actions are received
                win32event.SetEvent(self.action_received_event)

            elif result == win32event.WAIT_OBJECT_0 + 1:  # UpdateReadyEvent
                # For now, we do nothing for updates
                win32event.SetEvent(self.update_received_event)

if __name__ == "__main__":
    config = SharedMemoryAgentProcessorConfig(num_environments=5, num_actions=2, state_size=4, training_batch_size=10)
    agent_processor = SharedMemoryAgentProcessor(config)
    agent_processor.process()
