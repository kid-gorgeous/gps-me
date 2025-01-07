import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from trustee import ClassificationTrustee
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from argparse import ArgumentParser
arg = ArgumentParser()
# add names of the arguments for your user environment variables assuming
# the user has set them
user = os.getenv('SPACETRACKER_UNAME')
password = os.getenv('SP_PASSWORD')

def clear():
    import os
    os.system('clear')

arg.add_argument('--info', action='store_true', help='Display information about the file and dataset')
arg.add_argument('--train', action='store_true', help='Run training function')
# arg.add_argument('--use_trustee', action='store_true', help='Use the trustee model')


# Keras LSTM (Work in Progress)
class K_LSTM():
    def __init__(self, input_dim, hidden_dim, output_dim, *args, **kwargs):
        self.model = Sequential()
        self.model.add(Input(shape=(None, input_dim)))
        self.model.add(LSTM(input_dim, return_sequences=True, activation='tanh'))
        self.model.add(LSTM(hidden_dim, return_sequences=False, activation='tanh'))
        self.model.add(Dense(output_dim, activation='sigmoid'))
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
    
    def train(self, X_train, y_train):
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32)

    # @overload
    def predict(self, X_test):
        # prob_pred = self.model.predict(X_test)
        # rounded_pred = np.round(prob_pred)
        # predictions = tf.cast(rounded_pred, tf.int32)
        # return predictions.flatten()
        return self.model.predict(X_test)
    
    def compile(self):
        self.model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.001, rho=0.9))
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)

    def summary(self):
        print(self.model.summary())

    def save(self, filename):
        self.model.save(f'../models/{filename}')   

    def save_weights(self, filename):
        self.model.save_weights(f'../models/coefficients/{filename}')   


# Generate random data for X and y
X = np.random.randn(1000, 64)
# EPOCH
y = np.random.randn(1000, 1)

# Split the data into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Reshape the data for the LSTM model
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm = K_LSTM()

if __name__ == "__main__":
    args = arg.parse_args()
    if args.info:
        clear()
        print('This is a LSTM for the SpaceTracker API \n')
    elif args.train:
        print('Model trained successfully')
        # add and train dimensions here..
        # code was not included because this 
        # is used in another file
        print('Model evaluation: ')



