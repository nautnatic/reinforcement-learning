from pysc2.env import sc2_env
from pysc2.lib import features


class Env:
    def __init__(self, screen_size, minimap_size, game_steps_per_agent_step,
                 visualize=False):
        # set attributes
        self.screen_size = screen_size
        self.minimap_size = minimap_size
        self.game_steps_per_agent_step = game_steps_per_agent_step
        self.visualize = visualize

        # create env
        self.sc2_env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_size,
                                                       minimap=minimap_size),
                use_feature_units=True
            ),
            step_mul=game_steps_per_agent_step,
            visualize=visualize
        )

    def reset(self):
        """
        Resets the environment and returns the observation afterwards
        :return: Observation after the reset
        """
        return _preprocess_obs(self.sc2_env.reset())

    def step(self, action):
        """
        Executes an action in the environment.
        :param action: Action to execute
        :return: Observation of the environment after a agent step
        """
        return _preprocess_obs(self.sc2_env.step([action]))


def _preprocess_obs(timsteps):
    """
    Preprocess the observation
    :param timsteps:
    :return:
    """
    return timsteps[0]
