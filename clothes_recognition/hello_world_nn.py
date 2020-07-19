from tensorflow import keras
import numpy as np

if __name__ == '__main__':
    # Keras is a wrapper on tensorflow
    # Dense means a layer of connected neurons
    # units defines how many layers there are, hence there is one layer in this nn
    # input_shape is the shape of the input, for this nn there is one input
    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    # This defines the method for calculating the weights of this neural netword
    # loss is the method for defining the performance of the model, the error function
    # in this case we are using mean squared error
    # The optimizer is how the model improves itself, finds which way to move to reduce the error
    # in this case it is  sgd, or stochastic gradient descent
    model.compile(optimizer='sgd', loss='mean_squared_error')

    # use numpy to define the input array
    xs = np.array([-1, 0, 1,2, 3, 4], dtype=float)
    ys = np.array([-3, -1, 1, 3, 5, 7], dtype=float)

    # This finds the weights for the nn
    # Epochs defines how many guess, calculate error, optimize cycles there will be
    # In this case 500
    model.fit(xs, ys, epochs=500)

    print(model.predict([10.0]))
    # Why didn't it get the exact answer? First there is very little data, second the model has not find the optimal
    # weights yet, perhaps we need a different architecture for the neural network?
    # The neural network isn't positive, they deal in probabilites
