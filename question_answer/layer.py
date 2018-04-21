
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

        print("Timed inputs:", timed_inputs.shape)
        print("Fixed inputs:", fixed_inputs.shape)
        batch_size = tf.shape(timed_inputs)[0]
        print("Batch size:", batch_size)
        n_fixed_features = tf.shape(fixed_inputs)[-1]
        new_size = n_fixed_features * 32
        print("New size: ", new_size)

        n_timesteps = tf.shape(timed_inputs)[-2:-1]
        print("n_timesteps:", n_timesteps)
        tile_shape = [batch_size, 1, n_fixed_features]
        tile = tf.reshape(fixed_inputs, tile_shape)

        tile_multiples = (tf.constant(1), n_timesteps[0], tf.constant(1))

        tiled = tf.tile(tile, tile_multiples)

        combined_matrix = tf.concat((timed_inputs, tiled), axis=-1)

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
