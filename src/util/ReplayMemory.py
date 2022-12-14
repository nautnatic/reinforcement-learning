import numpy as np


class ReplayMemory:
    """
    Save transitions to do a batch update of the qvalues later
    """

    def __init__(self, min_size, max_size, minibatch_size):
        self.min_size = min_size
        self.max_size = max_size
        self.minibatch_size = minibatch_size
        self.memory = np.empty(shape=(0, 5))
        self.column_names = [
            'previous_state',
            'previous_action',
            'reward',
            'current_state',
        ]

    def get_random_minibatch(self):
        """
        Returns a random minibatch
        :return:
        """
        indices = np.random.choice(self.memory.shape[0],
                                   size=self.minibatch_size,
                                   replace=False)
        return self.memory[indices, :]

    def add(self, new_memory):
        """
        Adds a new memory to the replay memory
        :param new_memory: Memory to add
        """
        new_memory = np.array([(
                list(new_memory.previous_state),
                new_memory.previous_action,
                new_memory.reward,
                list(new_memory.current_state)
            )]
        )
        self.memory = np.vstack((self.memory, new_memory))
        if self.memory.shape[0] > self.max_size:
            self.memory = self.memory[1:, :]

    def min_size_reached(self):
        """
        Returns true if the minimum size was reached.
        :return:
        """
        return True if self.memory.shape[0] >= self.min_size else False


class MemoryEntry:
    """
    All attributes necessary in an entry of the replay memory
    """

    def __init__(self, previous_state, previous_action, reward, current_state):
        self.previous_state = previous_state
        self.previous_action = previous_action
        self.reward = reward
        self.current_state = current_state
