import datetime

from pathlib import Path

import tensorflow as tf
from tensorflow.python.summary.writer.writer import FileWriter


class Evaluator:
    def __init__(self, runner, output_container_dir):
        self.runner = runner
        self.output_container_dir = output_container_dir
        self.output_dir = None
        self.output_filename = None
        self.writer = None

    def initialize(self, ):
        """
        Initializes the evaluator in a new output directory
        """
        start_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        train_mode = ('train' if self.runner.train_mode else 'run')
        agent_name = type(self.runner.agent).__name__
        self.output_filename = f"{start_time}_{train_mode}_{agent_name}"
        self.output_dir = Path.joinpath(self.runner.output_container_dir, self.output_filename)
        self.writer = tf.compat.v1.summary.FileWriter(
            self.output_dir,
            tf.compat.v1.get_default_graph()
        )

    def add_summary(self, title, x_value, y_value):
        """
        Adds a summary to the event file
        :param title: Title of the diagram
        :param x_value: x value (usually episode)
        :param y_value: y value (the value to visualize)
        :return:
        """
        if self.writer is None:
            raise Exception("Evaluator wasn't initialized")

        self.writer.add_summary(tf.compat.v1.Summary(
            value=[tf.compat.v1.Summary.Value(tag=title, simple_value=y_value)]),
            x_value
        )


