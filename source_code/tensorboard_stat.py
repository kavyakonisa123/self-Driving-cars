"""
AUTHOR: Was taken from https://github.com/Noam2710/deep-q-learning/blob/master/ModifiedTensorBoard.py
FILENAME: tensorboard_stat.py
SPECIFICATION: In this file, we have used the tensorboard concept to save logs, update stats to visualize the logds in tensboard later
FOR: CS 5392 Reinforcement Learning Section 001
"""
import tensorflow as tf
from keras.callbacks import TensorBoard

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.FileWriter(self.log_dir)

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass
