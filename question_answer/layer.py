
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow as tf
from question_answer.util import sigmoid, logit
#

class ScaleMaxSigmoid(Layer):

    def __init__(self, scale_max_to=0.95, **kwargs):
        self.scale_max_to = tf.constant(scale_max_to)
        super(ScaleMaxSigmoid, self).__init__(**kwargs)

    def logit(self, tensor):
        return tf.log(tensor/(tf.constant(1.0)-tensor))

    def call(self, inputs):
        # return inputs
        mx = tf.reduce_max(inputs)
        ratio = self.scale_max_to / tf.sigmoid(mx)
        # logit_bonus = self.logit(self.scale_max_to) - self.logit(tf.sigmoid(mx))

        return tf.sigmoid(inputs) * ratio

    #
    # def call(self, inputs):
    #     # return inputs
    #     mx = tf.reduce_max(inputs)
    #     logit_bonus = self.logit(self.scale_max_to) - self.logit(tf.sigmoid(mx))
    #
    #     return inputs + logit_bonus


class ContextRepeat(Layer):

    def __init__(self, **kwargs):
        super(ContextRepeat, self).__init__(**kwargs)

    def call(self, inputs):
        timed_inputs, fixed_inputs = inputs
        n_repeats = tf.shape(timed_inputs)[0:1]

        # print("Timed inputs:", timed_inputs.shape)
        # print("Fixed inputs:", fixed_inputs.shape)
        # print("Tf timed inputs:", tf.shape(timed_inputs))
        # print("N repeats", n_repeats, n_repeats.shape)
        #

        print("Timed inputs:", timed_inputs.shape)
        print("Fixed inputs:", fixed_inputs.shape)
        batch_size = K.shape(timed_inputs)[0]
        print("Batch size:", batch_size)
        n_fixed_features = K.shape(fixed_inputs)[-1]
        new_size = n_fixed_features * 32
        print("New size: ", new_size)

        n_timesteps = K.shape(timed_inputs)[-2:-1]
        print("n_timesteps:", n_timesteps)
        tile_shape = [batch_size, 1, n_fixed_features]
        tile = K.reshape(fixed_inputs, tile_shape)

        tile_multiples = (tf.constant(1), n_timesteps[0], tf.constant(1))

        tiled = K.tile(tile, tile_multiples)

        combined_matrix = K.concatenate((timed_inputs, tiled), axis=-1)

        new_shape = [32, 104, 64]
        print("new_shape", new_shape)
        print("matrix shape:", combined_matrix.shape)
        # fixed_inputs = tf.reshape(
        #     fixed_inputs, shape=(-1, n_fixed_features))
        # print(fixed_inputs.shape)
        # tiled = tf.tile(fixed_inputs, n_timesteps)
        # # not really tiled so much as appended into a long list
        #
        # new_shape = [batch_size, n_timesteps[0], n_fixed_features]
        # # get the shape to convert it to
        #
        # matrix = tf.reshape(tiled, new_shape)
        # # now lay it down in true "tiles"

        print(combined_matrix.shape)
        # put alongside the timestep-specific vectors

        return combined_matrix


    def compute_output_shape(self, input_shape):

        timed_shape, fixed_shape = input_shape
        n_output_features = timed_shape[-1] + fixed_shape[-1]
        return (
            timed_shape[0],
            timed_shape[1],
            n_output_features)
