
from keras.engine.topology import Layer
import tensorflow as tf

#
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

        n_fixed_features = tf.shape(fixed_inputs)[-1]
        fixed_inputs = tf.reshape(
            fixed_inputs, shape=(n_fixed_features,))

        n_timesteps = tf.shape(timed_inputs)[-2:-1]
        tiled = tf.tile(fixed_inputs, n_timesteps)
        # not really tiled so much as appended into a long list

        new_shape = [1, n_timesteps[0], n_fixed_features]
        # get the shape to convert it to

        matrix = tf.reshape(tiled, new_shape)
        # now lay it down in true "tiles"

        combined_matrix = tf.concat((timed_inputs, matrix), axis=-1)
        # put alongside the timestep-specific vectors

        return combined_matrix


    def compute_output_shape(self, input_shape):

        timed_shape, fixed_shape = input_shape
        n_output_features = timed_shape[-1] + fixed_shape[-1]
        return (
            timed_shape[0],
            timed_shape[1],
            n_output_features)
