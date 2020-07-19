import tensorflow as tf
from tensorflow import keras

if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist
    # We are splitting the data into a training set and a test set
    # So we can see how god our neural network is on examples it hasn't seen before
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    # In this case the input will be picture of size 28,28 but we want to put it through one layer
    # So here we use keras.layers.Flatten to flatten an input of the shape (28,28)
    # Now we create a Dense of size 128 to do some calculations in the middle, this is the hidden layer
    # And an output dense of size 10 as there are 10 output categories (shirt, hat, etc)
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
