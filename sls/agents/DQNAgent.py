import numpy as np
from keras import Sequential, Input, activations, initializers
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.layers import Dense, Rescaling

from sls.agents import AbstractAgent
from sls.util.EpsilonGreedyExploration import EpsilonGreedyExplorationStrategy

from sls.util.History import History
from sls.util.ReplayMemory import ReplayMemory
from sls.util.networks import Network


class DQNAgent(AbstractAgent):
    def __init__(self, runner, pretrained_model=None):
        super(DQNAgent, self).__init__(screen_size=runner.env.screen_size)

        # validate input
        if not runner.train_mode and pretrained_model is None:
            raise Exception(
                "A model needs to be provided in run mode. Either provide a model or run in train mode.")

        # set attributes
        self.runner = runner
        # discounting factor. Determines how much a terminal state should propagate
        # to the previous state. (in range [0,1])
        self.gamma = 0.9
        self.exploration_strategy = EpsilonGreedyExplorationStrategy(
            config=self,
            epsilon_start=1,
            epsilon_end=0.05,
            episodes_until_min_epsilon=500)
        self.replay_memory = ReplayMemory(
            min_size=6000,
            max_size=10000,
            minibatch_size=32)
        self.agent_step_history = History()

        if pretrained_model is not None:
            model = pretrained_model
        else:
            model = build_model(env=self.runner.env, learning_rate=0.00025)
        # the network is a wrapper for the model, which contains methods to manage the model
        self.network = Network(model)
        self.target_network = self.network.clone()

        if runner.train_mode:
            model.trainable = False

    def save(self):
        """
        Saves the trained model.
        """
        self.network.save_model(output_dir=self.runner.evaluator.output_dir)

    def write_summary(self):
        self.runner.evaluator.add_summary(
            title='Score per Episode',
            x_value=self.runner.score,
            y_value=self.runner.current_episode
        )

        if len(self.runner.episode_history.size) >= 50:
            average_score = np.average(self.runner.episode_history.value[-50:])
            self.runner.evaluator.add_summary(
                title='Average score (last 50 episodes)',
                x_value=average_score,
                y_value=self.runner.current_episode
            )

    def step(self, observation):
        """
        Executes an agent step
        :param observation: Observation of the current situation
        :return: SC2 action that should be executed in the environment
        """
        # TODO What does this do?
        if not (self._MOVE_SCREEN.id in observation.observation.available_actions):
            return self._SELECT_ARMY

        marine_coords, beacon_coords = self.get_coordinates(observation)
        # TODO: convert to tuple?
        # state = marine coordinates converted into beacon-centered coordinate system
        state = marine_coords - beacon_coords
        next_action = self.exploration_strategy.get_next_action(state)

        if self.runner.train_mode:
            self._train_step(observation, state, next_action)

        next_sc2_action = self._dir_to_sc2_action(next_action, marine_coords)
        return next_sc2_action

    def _train_step(self, observation, current_state, next_action):
        """
        Executes the part of a step only relevant for training
        :param observation:
        """
        # create new memory entry
        if not self.agent_step_history.is_empty():
            new_memory = MemoryEntry(
                previous_state=self.agent_step_history.entries[-1].state,
                previous_action=self.agent_step_history.entries[-1].action,
                reward=self.agent_step_history.entries[-1].reward,
                current_state=current_state)
            self.replay_memory.add(new_memory)

        # update network with minibatch
        if self.replay_memory.min_size_reached():
            batch = self.replay_memory.get_random_minibatch()
            target_prediction = self.predict(batch)
            self.network.execute_gradient_descent(batch, target_prediction)

        # update agent step history
        if observation.last() or observation.reward == 1:
            self.agent_step_history.reset()
        else:
            self.agent_step_history.add((next_action, current_state))

        # update exploration rate
        if observation.last():
            self.exploration_strategy.update_exploration_rate()

    def get_coordinates(self, observation):
        """
        Gets marine and beacon positions in initial coordinate system
        :param observation: Obervation of the environment
        :return: tuple: (marine_coords, beacon_coords)
        """
        marine = self._get_marine(observation)
        beacon = self._get_beacon(observation)
        marine_coords = self._get_unit_pos(marine)
        beacon_coords = self._get_unit_pos(beacon)
        return marine_coords, beacon_coords

    def predict(self, batch):
        """
        Returns the predictions for all elements of the batch
        :param batch:
        :return:
        """
        # 1. predict all actions for the current state in the current model
        y_target = self.target_network.model.predict(np.stack(batch[:, 3]))
        # 2. get best action A_t (highest predicted reward) for the current
        # target model
        y_pred = self.network.model.predict(np.stack(batch[:, 0]))

        # 3. for best action: update q values of target model according to:
        # q_target = r + gamma * max_a(q)
        for sample_nb, sample in enumerate(batch):
            beacon_reached = True if sample.reward == 1.0 else False
            index_executed_action = list(self._DIRECTIONS.keys()).index(
                sample[1])
            if beacon_reached:
                y_executed_action = sample.reward
            else:
                y_executed_action = \
                    (sample.reward + self.gamma * np.max(y_target[sample_nb]))
            y_pred[sample_nb, index_executed_action] = y_executed_action
        return y_pred


class MemoryEntry:
    """
    All attributes necessary in an entry of the replay memory
    """
    def __init__(self, previous_state, previous_action, reward, current_state):
        self.previous_state = previous_state
        self.previous_action = previous_action
        self.reward = reward
        self.current_state = current_state


def build_model(env, learning_rate):
    """
    Creates neural network model
    :param env: Injected Environment reference
    :param learning_rate: Learning rate used in the network
    :return: Compiled model
    """
    model = Sequential()
    # input layer defines shape (2,0) -> 2D for x,y
    model.add(
        Input(shape=(2,)))
    # scale input to interval [-1;1]
    model.add(
        Rescaling(scale=1 / (env.screen_size / 2), offset=-1))
    model.add(
        Dense(16, activation=activations.relu,
              kernel_initializer='random_normal'))
    model.add(
        Dense(32, activation=activations.relu,
              kernel_initializer='random_normal'))
    model.add(
        Dense(8, activation=activations.linear,
              kernel_initializer='random_normal'))
    return model.compile(optimizer=Adam(learning_rate=learning_rate),
                         loss=MeanSquaredError())
