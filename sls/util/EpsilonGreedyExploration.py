class EpsilonGreedyExplorationStrategy:
    def __init__(self, config, epsilon_start, epsilon_end,
                 episodes_until_min_epsilon):
        """
        :param config: Reference to the Agent that uses the exploration strategy
        :param epsilon_start: Exploration rate at the beginning. In range [0,1]
        :param epsilon_end: Exploration from episodes_until_min_epsilon until
        the end. In range [0,1]
        :param episodes_until_min_epsilon: Number of episodes until the minimum
        exploration rate is reached.
        """
        self.config = config
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_current = epsilon_start
        self.episodes_until_min_epsilon = episodes_until_min_epsilon

    def get_next_action(self, current_state):
        """
        Returns next action according to epsilon greedy strategy
        :param current_state: State to take the action from
        :return: Next action to execute.
        """
        if self.config.train_mode and (
                np.random.uniform(0, 1) < self.epsilon_current):
            # choose random action
            return np.random.choice(list(self.config.DIRECTIONS.keys()))

        # choose action with the highest reward
        input = np.expand_dims(list(current_state), axis=0)
        return list(self.config.DIRECTIONS.keys())[
            np.argmax(self.config.model.predict(input))]

    def update_exploration_rate(self):
        """
        Updates epsilon linearly decreasing according to epsilon_start,
        epsilon_end and episodes_until_min_epsilon
        :return:
        """
        if self.epsilon_current > self.epsilon_end:
            self.epsilon_current -= (
                                            self.epsilon_start - self.epsilon_end) / self.episodes_until_min_epsilon
