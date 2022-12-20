from absl import app
from sls.agents.DQNAgent import DQNAgent
from sls.runtime import Env, Runner


def main(unused_argv):
    env = Env(
        screen_size=64,
        minimap_size=64,
        visualize=True,
        game_steps_per_agent_step=8
    )
    runner = Runner(
        agent_class=DQNAgent,
        env=env,
        nb_episodes=100,
        output_container_dir='./out/...'
    )
    runner.run(pretrained_model_path='./out/FINISHED_train_DQNAgent/model.hdf5')


if __name__ == "__main__":
    app.run(main)
