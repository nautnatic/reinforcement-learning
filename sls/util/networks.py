from pathlib import Path

import numpy as np
from abc import ABC

import tensorflow as tf


class Network(ABC):
    def __init__(self, model):
        self.model = model

    def save_model(self, output_dir, file_name="model.hdf5"):
        """
        Saves the contained model as a hdf5 file
        :param output_dir: Directory in which the file gets created
        :param file_name: Name of the hdf5 file that should be created
        """
        file_path = Path.joinpath(output_dir, file_name)
        self.model.save(file_path)

    def execute_gradient_descent(self, batch, target_prediction):
        """
        Execute Gradient Descent step
        :param batch:
        :param target_prediction:
        :return:
        """
        self.model.fit(x=np.stack(batch[:, 0]), y=target_prediction,
                       epochs=1, verbose=0)

    def clone(self):
        """
        Returns a clone of this network
        :return: Clone of the network
        """
        new_model = self.model.clone_model(self.model)
        return Network(new_model)


def load_pretrained_model(path):
    """
    Loads a model from a hdf5 file path
    :param path: hdf5 file path
    :return: Loaded model
    """
    return tf.keras.models.load_model(path)
