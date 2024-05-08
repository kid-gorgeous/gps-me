import os

import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input

from argparse import ArgumentParser
arg = ArgumentParser()


def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')


arg.add_argument('--info', action='store_true', help='Display information about the file and dataset')
arg.add_argument('--test', action='store_true', help='Run the main function')
arg.add_argument('--run_train', action='store_true', help='Run the training function')


# Keras RNN
class K_RNN(keras.Model):

    # Initialize the model
    # Expect 3 dimensions: input_dim, hidden_dim, output_dim
    def __init__(self, input_dim, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.model = Sequential()
        self.model.add(Input(shape=(None, input_dim)))
        self.model.add(SimpleRNN(hidden_dim))
        self.model.add(Dense(output_dim))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=5, batch_size=32)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)

    def info(self):
        model = self.model
        print(model.summary())
        print("\nModel layers: ", model.layers)
        print("Data shapes (X,y): ",X.shape, y.shape)

# - - - - #
# Generate random data for X and y
X = np.random.randn(1000, 64)
y = np.random.randn(1000, 1)

# Split the data into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Reshape the data for the LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

if __name__ == "__main__":
    args = arg.parse_args()
    rnn = K_RNN(64, 16, 1)
    y_pred = None

    if args.info:
        clear_terminal()
        rnn.info()


    elif args.run_train:
        rnn.train(X_train, y_train)

        y_pred = rnn.predict(X_test)
        print("Model evaluation: ", rnn.evaluate(X_test, y_test))
    
    
    else:
        print('No arguments provided. Please use the --help for more information. \n')
        

