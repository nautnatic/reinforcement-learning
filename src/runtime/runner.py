import os
import tensorflow as tf

from src.util.History import History
from src.util.evaluation import Evaluator
from src.util import networks

tf.compat.v1.disable_eager_execution()


class Runner:
    """
    Resposible for running/training the agent in a given environment
    """
    def __init__(self, agent_class, env, nb_episodes, output_container_dir):
        """
        :param agent_class: Agent navigation in the environment
        :param env: Environment that the agent operates in
        :param nb_episodes: Number of episodes this runner is run.
        :param output_container_dir: File path to the directory which contains
        output directories for all runs. The run output directories contain for
        example the trained model, that can be loaded later.
        """
        # validate input
        if output_container_dir is None or not os.path.isdir(output_container_dir):
            raise Exception(f"Invalid value for output directory {output_container_dir}")

        self.output_container_dir = output_container_dir
        self.agent_class = agent_class
        self.env = env
        self.nb_episodes = nb_episodes
        self.evaluator = Evaluator(runner=self, output_container_dir=output_container_dir)

        self.episode_history = None
        self.current_episode = None
        self.score = None
        self.train_mode = None
        self.agent = None

    def train(self):
        """
        Execute the runner in training mode
        """
        self.train_mode = True
        self.agent = self.agent_class(runner=self)
        self._exec()

    def run(self, pretrained_model_path):
        """
        Execute the runner in normal mode, which uses a pretrained network
        :param pretrained_model_path: Run name whose network weights are used as a source.
        """
        self.train_mode = False
        model = networks.load_pretrained_model(pretrained_model_path)
        self.agent = self.agent_class(runner=self, pretrained_model=model)
        self._exec()

    def _exec(self):
        """
        Execute this runner as configured before. This method is used by both
        the run and the train method.
        """
        # common setup for run and train
        self.episode_history = History(size=50)
        self.evaluator.initialize()

        for episode in range(1, self.nb_episodes):
            # episode setup
            self.current_episode = episode
            observation = self.env.reset()
            self.score = 0  # reset score

            # execute agent steps until limit is reached
            while True:
                action = self.agent.step(observation)
                if observation.last():
                    break
                observation = self.env.step(action)
                self.score += observation.reward

            # finish episode (write summary and save model data)
            self.episode_history.add(self.score)
            self._write_summary()
            if self.train_mode and episode % 10 == 0:
                self._save_model()
                try:
                    self.agent.update_target_model()
                except AttributeError:
                    ...

    def _write_summary(self):
        """
        Writes all summaries
        """
        self.agent._write_summary()

    def _save_model(self):
        """
        Saves all models
        """
        self.agent.save()
