
import numpy as np
from collections import defaultdict
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Activation
from keras.models import Model


def fake_x_y(n_observations=100, n_inputs=3, n_outputs=2, noise_rate=0.1):
    possible_operators = [np.add, np.multiply, np.divide]
    output_transforms = defaultdict(list)
    for output in range(n_outputs):
        for _ in range(0, n_inputs):
            # get operators to interleave between inputs to yield an output for
            # the nn to search for
            output_transforms[output].append(np.random.choice(possible_operators))
    x = np.random.normal(size=(n_observations,n_inputs))
    output_ys = []

    for output in range(n_outputs):
        output_y = x[:, 0]
        for idx, transform in enumerate(output_transforms[output]):
            output_y = transform(output_y, x[:, idx])

        output_y = add_noise(output_y, rate=noise_rate)
        output_ys.append(output_y.reshape((n_observations,1)))


    y = np.concatenate(output_ys, axis=1)
    return x, y



def simple_mlp(
        n_inputs=3, n_outputs=2, neuron_counts=[8,12],
        activation='relu', final_activation='linear',
        optimizer='adam', loss='mse',
        kernel_initializer='truncated_normal'):

    inputs = Input(shape=(n_inputs,))
    layer = inputs
    activations = ([activation]*(len(neuron_counts)-1))+[final_activation]
    for nc, act in zip(neuron_counts, activations):
        layer = Dense(
            nc, activation=act,
            kernel_initializer=kernel_initializer)(layer)
    outputs = Dense(
        n_outputs, activation=final_activation,
        kernel_initializer=kernel_initializer)(layer)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss=loss)
    return model, inputs, outputs



def add_noise(var, rate=0.1):
    return var + rate*(np.random.normal(size=var.shape)-var)



def quick_scatterplot(x, y):
    df = pd.DataFrame(np.concatenate([x, y], axis=1), columns=['x', 'y'])
    ax = sns.regplot(x='x', y='y', data=df)
    plt.show()


n_inputs = 1
n_outputs = 1
x, y = fake_x_y(
    n_observations=500,
    n_inputs=n_inputs,
    n_outputs=n_outputs,
    noise_rate=0.1)
quick_scatterplot(x, y)

model, _,_ = simple_mlp(n_inputs=n_inputs, n_outputs=n_outputs)

for n in range(10):
    model.fit(x, y, validation_split=0.2)


