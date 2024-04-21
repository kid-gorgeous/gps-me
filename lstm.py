import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from argparse import ArgumentParser
arg = ArgumentParser()

from api import SP_Client

user = os.getenv('SPACETRACKER_UNAME')
password = os.getenv('SP_PASSWORD')

sp = SP_Client(user, password)
sp.set_data()

line_data_1 = sp.tle_df_line1
line_data_2 = sp.tle_df_line2

arg.add_argument('--info', action='store_true', help='Display information about the file and dataset')
arg.add_argument('--train', action='store_true', help='Run training function')

# Keras LSTM (Work in Progress)
class K_LSTM:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.model = Sequential()
        self.model.add(Input(shape=(None, input_dim)))
        self.model.add(LSTM(input_dim, return_sequences=True))
        self.model.add(LSTM(hidden_dim))
        self.model.add(Dense(output_dim))

        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train, epochs=10, batch_size=32)

    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)
    
    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = keras.models.load_model(filepath)

    def summary(self):
        print(self.model.summary())
        
from sklearn.linear_model import train_test_split



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


if __name__ == "__main__":
    args = arg.parse_args()
    lstm = K_LSTM(64, 16, 1)
    y_pred = None

    if args.info:
        print('This is a LSTM for the SpaceTracker API \n')
        lstm.summary()
    elif args.train:
        lstm.train(X_train, y_train)
        y = lstm.predict(X_test)
        
        print('Model trained successfully')
        print('Model evaluation: ', lstm.evaluate(X_test, y_test))

    else:
        pass

