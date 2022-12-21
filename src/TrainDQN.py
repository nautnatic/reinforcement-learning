from absl import app
from src.agents.DQNAgent import DQNAgent
from src.runtime import Env, Runner


def main(unused_argv):
    env = Env(
        screen_size=64,
        minimap_size=64,
        visualize=False,
        game_steps_per_agent_step=8
    )
    runner = Runner(
        agent_class=DQNAgent,
        env=env,
        nb_episodes=1000,
        output_container_dir='./out'
    )
    runner.train()


if __name__ == "__main__":
    app.run(main)
