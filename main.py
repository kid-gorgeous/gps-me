'''
This script downloads data from space-track.org. 
'''


import time
from termcolor import colored

from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import tensorflow as tf

# Import spacetrack library modules
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
import matplotlib.pyplot as plt
from datetime import datetime

# Import auxiliary libraries
import datetime as dt
import pandas as pd
import numpy as np
import time as tm
import json as js
import threading
import httpx
import sys

from sklearn.metrics import confusion_matrix
import seaborn as sns

import os


# Credentials to access the Space-track API
identity = os.environ.get('SPACETRACKER_UNAME')
password = os.environ.get('SP_PASSWORD')
expr = colored(f'{identity}', 'yellow')
print(f'Logging in as: {expr}')

# List of satellite NORAD CAT IDs
ids = [
    41335,  # Sentinel-3A
    43437,  # Sentinel-3B
    25041,  # Jason-1
    33105,  # Jason-2
    41335,  # Jason-3 (Note: Same as Sentinel-3A)
    36508,  # Cryosat-2
    27386,  # Envisat
    22076   # TOPEX/Poseidon
    
]

# Function to print the runtime of a function
def print_runtime(start, end, function_id):
    print(f'Time taken: {end-start} for function class {function_id}')

def parse_data(id, omm_data, filepath):
    function_id = 'parse_data'
    # Parse the data and save into a dataframe
    omm = js.loads(omm_data)
    df  = pd.DataFrame(omm)
    fileName = filepath + f'/sat-data{id}.csv'
    df.to_csv(fileName, mode='a', index=False)

def get_data(filepath):
    time = tm.time()
    function_id = 'get_data'
    print(f'Logging in as {identity}')

    # Initialize the spacetrack client object
    cl = httpx.Client(timeout = 300)
    st = SpaceTrackClient(identity, password, httpx_client=cl)
    print(f'Client initialized: {cl}, SpaceTrackClient: {st}')

    # Set the date range for the requested data
    drange = op.inclusive_range(dt.datetime(2019, 3, 1), dt.datetime(2020, 3, 1))

    # Build a list with all NORAD CAT IDs. They are consecutive starting with 1 and ending
    win_size = 100

    fileName  = 'sat-data.csv'
    # Iterate through all satellite IDs and retrieve TLEs within the date range
    for id in ids:
        print(f'Processing NORAD CAT ID: {id}')
        # Build list of ids to query
        id_list = list(range(id,id+win_size))
        # start runtime 
        time = tm.time()
        print("Sending request...")
        omm = st.omm(norad_cat_id=id_list , format='json', epoch=drange)
        parse_data(id, omm, filepath)
        print("Completed request")

    end = tm.time()
    print_runtime(time, end, function_id)

def plot_data(df):
    function_id = 'plot_data'
    print(f'Plotting the data...')

    features = df.columns.drop('EPOCH')  # Assuming 'epoch' is the name of your epoch column

    for feature in features:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['EPOCH'], df[feature])
        plt.xlabel('Epoch')
        plt.ylabel(feature)
        plt.title(f'{feature} over Epochs')
        plt.show()

    plt.show()

# Open the file and read the data from the file path, and return the dataframe
def open_file(filepath):
    start = tm.time()
    function_id = 'open_file'
    large_df = pd.DataFrame()
    for i in range(len(ids)):
        sat_id = ids[i]
        filename = filepath + f'/sat-data{ids[i]}.csv'
        term = colored(f'Opening {filename}', 'green')
        data = pd.read_csv(filename, dtype=str, delimiter=',')

        # EPOCK: 10, 
        EPOCH = data.columns[10]
        # Inclination: 13
        INCLINATION = data.columns[13]
        # RAAN: 14
        RAAN = data.columns[14]
        # Mean anomaly: 16
        MEAN_ANOMALY = data.columns[16]
        # Semi-major axis: 28
        SEMIMAJOR_AXIS = data.columns[28]
        data_cols = ['SAT_ID',EPOCH, INCLINATION, RAAN, MEAN_ANOMALY, SEMIMAJOR_AXIS]

        # Create a new dataframe with the selected columns
        df = pd.DataFrame(data, columns=data_cols)
        df['SAT_ID'] = sat_id
        df['EPOCH'] = pd.to_datetime(df['EPOCH'], format='mixed', errors='coerce')
        df['EPOCH'] = pd.to_datetime(df['EPOCH'], unit='s')
        df['EPOCH'] = df['EPOCH'].astype(int) / 10**9
        df['EPOCH'] = df['EPOCH'].astype('datetime64[s]').astype(int)

        # Convert non-numeric values to NaN
        df['INCLINATION'] = pd.to_numeric(df['INCLINATION'], errors='coerce')

        # Then, you can convert the column to floats
        df['INCLINATION'] = df['INCLINATION'].astype(float)


        # Calculate the difference between the columns
        df_diff = df.copy()
        df_diff['EPOCH'] = pd.to_datetime(df_diff['EPOCH'], unit='s')
        df_diff['EPOCH'] = df_diff['EPOCH'].astype(int) / 10**9
        df_diff['EPOCH'] = df_diff['EPOCH'].astype('datetime64[s]').astype(int)
        df_diff.columns = ['D_' + col for col in data_cols]
        
        # Concatenate df and df_diff
        df_combined = pd.concat([df, df_diff], axis=1)
        df_combined = df_combined.dropna()
        df_combined = df_combined.drop(['D_SAT_ID'], axis=1)
        df_combined = df_combined.reset_index(drop=True)
        df_combined['EPOCH'] = pd.to_datetime(df_combined['EPOCH'], unit='s')
        df_combined['EPOCH'] = df_combined['EPOCH'].astype(int) / 10**9
        df_combined['EPOCH'] = df_combined['EPOCH'].astype('datetime64[s]').astype(int)
        df_combined['EPOCH_diff'] = df_combined['EPOCH'].diff()
            
        # Classify EPOCH_diff
        # Define the threshold for a maneuver of 100 seconds
        maneuver_threshold = pd.to_timedelta('100 seconds').total_seconds()
        df_combined['maneuver_made'] = (df_combined['EPOCH_diff'] > maneuver_threshold).astype(int)

        # Create a new binary column that indicates whether a maneuver was made
        large_df = pd.concat([large_df, df_combined], axis=0)
        print_runtime(start, tm.time(), function_id)
    return large_df

def calculate_diff(df):
    function_id = 'calculate_diff'
    print(f'Calculating the difference between the columns, and change the col names...')

    timestamp_str = "2019-03-01T11:47:54.304800"
    timestamp = datetime.fromisoformat(timestamp_str)

    # Convert the EPOCH column to datetime
    df['EPOCH'] = pd.to_datetime(df['EPOCH'], format='mixed', errors='coerce')
    print(timestamp.dt.total_seconds())
    
    # Convert the columns to numeric
    for col in df.columns:
        if col != 'EPOCH':
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df_diff = df.diff()
    return df_diff

def _calculate_diff(df):
    function_id = 'calculate_diff'
    print(f'Calculating the difference between the columns...')

    # Convert the EPOCH column to datetime
    df['EPOCH'] = pd.to_datetime(df['EPOCH'], errors='coerce')

    # Calculate the difference between the timestamps
    df['EPOCH_diff'] = df['EPOCH'].diff()

    # Convert the time differences to seconds
    df['EPOCH_diff'] = df['EPOCH_diff'].dt.total_seconds()
    return df

def train(obj):
    try:
        df = open_file(filepath)
        # Assuming df_combined is your DataFrame and 'target' is your class label
        print('Training of model...')
        
        X = df.drop(['maneuver_made','EPOCH_diff', 'D_EPOCH'], axis=1)
        X = X.to_numpy().reshape((X.shape[0], 1, X.shape[1]))
        y = df['maneuver_made']
        
        print('Splitting the data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f'Shapes of: {X_train.shape, y_train.shape, X_test.shape, y_test.shape}')

        print('Encoding the labels...')
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        print('Training the model...')
        X_train = X_train.astype(np.float32)
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        obj.train(X_train, y_train)

        print('Predicting the model...')
        X_test = X_test.astype(np.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        X_test = tf.cast(X_test, dtype=tf.float32)

        y_pred = obj.predict(X_test)

        print('Evaluating the model...')
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        X_test = tf.cast(X_test, dtype=tf.float32)
        obj.evaluate(X_test, y_test)

        print('Optimizing the model...')
        obj.compile()

        print('Saving the model...')
        # Save the model in the H5 format
        obj.save('test_primative_model.keras')
        obj.save_weights('test_primative.weights.h5')

        # Save training data
        np.save('../models/X_train.npy', X_train)
        np.save('../models/y_train.npy', y_train)

        # # Save testing data
        np.save('../models/X_test.npy', X_test)
        np.save('../models/y_test.npy', y_test)

        # # Save predictions
        np.save('../models/y_pred.npy', y_pred)
        
        print('Model trained successfully')

        obj.summary()


        # Assume y_pred are the predicted probabilities or scores
        y_pred = (y_pred > 0.5).astype(int)

        # Assume y_test are the true values and y_pred are the model's predictions
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 Score: {f1}')
        print(f'ROC AUC: {roc_auc}')


        # Assume y_test are the true values and y_pred are the model's predictions
        cm = confusion_matrix(y_test, y_pred)
        y_score = obj.predict(X_test)
        # Assume y_test are the true values and y_score are the predicted probabilities
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    except Exception as e:
        expr = colored('Exception in main(): ', 'red')
        print(f'{expr}', str(e))

def load_mod():
    try:
        print('Loading the model...')
        model = load_model('../models/model.keras')
        print('Model loaded successfully')
        # Assume model is a new instance of your Keras model
        weights = model.load_weights('../models/primative.weights.h5')   
        model.summary()

        # Load training, testing data, and predictions
        X_train = np.load('../models/X_train.npy')
        y_train = np.load('../models/y_train.npy')
        X_test = np.load('../models/X_test.npy')
        y_test = np.load('../models/y_test.npy')
        y_pred = np.load('../models/y_pred.npy')

        print('Loading and Evaluating model...')
        df = open_file(filepath)
        X = df.drop(['maneuver_made','EPOCH_diff', 'D_EPOCH'], axis=1)
        X = X.to_numpy().reshape((X.shape[0], 1, X.shape[1]))
        y = df['maneuver_made']

        print('Splitting the data...')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        print(f'Shapes of: {X_train.shape, y_train.shape, X_test.shape, y_test.shape}')

        print('Encoding the labels...')
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

        print('Fitting the model...')
        # Create an EarlyStopping callback
        # early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        # Convert the training and testing data to float32 type
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)

        # Convert the training and testing data to TensorFlow tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)

        # Fit the model to the training data
        history = model.fit(X_train, y_train, epochs=100, 
                            validation_split=0.2)

        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        # Generate predictions for the testing data using the model
        y_pred = model.predict(X_test)
        y_pred = y_pred.round().astype(int)
        
        # Evaluate the model on the testing data
        results = model.evaluate(X_test, y_test, verbose=0)

        # Save the model and the weights
        model.save('fine_tuned_model.keras')
        model.save_weights('fine_tuned.weights.h5')

        # If the model has only one metric (e.g., accuracy), results will be a single float
        if isinstance(results, float):
            accuracy = results
            print(f'Test accuracy: {accuracy:.3f}')
        # If the model has more than one metric, results will be a list
        else:
            loss, accuracy = results
            print(f'Test loss: {loss:.3f}')
            print(f'Test accuracy: {accuracy:.3f}')
 
    except Exception as e:
        expr = colored('Exception in main(): ', 'red')
        print(f'{expr}', str(e))

        
def clever_attack(model):
    from cleverhans.future.tf2.attacks import fast_gradient_method

    # Assume model is your trained Keras model
    # Assume X_test and y_test are your testing data
    X_test = np.open('../models/X_test.npy')
    y_test = np.open('../models/y_test.npy')

    # Convert the testing data to TensorFlow tensors
    X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

    # Define the loss function
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Generate adversarial examples
    X_test_adv = fast_gradient_method.fast_gradient_method(model, X_test_tensor, 0.3, np.inf, targeted=False)

    # Evaluate the model on the adversarial examples
    loss, acc = model.evaluate(X_test_adv, y_test_tensor, verbose=0)
    print(f'Adversarial test accuracy: {acc:.3f}')

def use_trustee(lstm):
    
        from trustee import ClassificationTrustee
        model = load_model('../models/test_primative_model.keras')

        X_train = np.load('../models/X_train.npy')
        y_train = np.load('../models/y_train.npy')
        X_test = np.load('../models/X_test.npy')
        y_test = np.load('../models/y_test.npy')
        y_pred = np.load('../models/y_pred.npy')
        
        clear()
    
        lstm.summary()
        print('Using trustee...')
        print(f'Shapes of: {X_train.shape, y_train.shape, X_test.shape, y_test.shape}')

        trustee = ClassificationTrustee(expert=lstm)

        X_train_2d = X_train.reshape(X_train.shape[0],-1)
        # X_train_reshaped = np.reshape(X_train, (-1, 1, 10))

        # X_train_2d = np.reshape(X_train_reshaped, (X_train_reshaped.shape[0], -1))
        # Add a time step dimension to X_train_2d
        X_train_3d = np.expand_dims(X_train_2d, axis=1)
    
        trustee.fit(X_train_2d, y_train, predict_method_name='predict_trustee', num_iter=100, num_stability_iter=10, samples_size=0.3, verbose=True)
        dt, pruned_dt, agreement, reward = trustee.explain()
        dt_y_pred = dt.predict(X_test.reshape(X_test.shape[0], -1))

        print("Model explanation global fidelity report:")
        print(classification_report(y_pred, dt_y_pred))
        print("Model explanation score report:")
        print(classification_report(y_test, dt_y_pred))

# Start the timer and run the main function
start  = tm.time()
if __name__ == '__main__':
    filepath = '../data/sat-data'
    modelpath = '../models/model.keras'
    from argparse import ArgumentParser

    from lstm import K_LSTM
    from rnn import K_RNN
    rnn = K_RNN(10, 16, 1)
    lstm = K_LSTM(10, 16, 1)

    df = open_file(filepath)

    arg = ArgumentParser()
    arg.add_argument('--upath',action='store_true', help='Using saved data')
    arg.add_argument('--dl', action='store_true',help='Download data')
    arg.add_argument('--lstm_info', action='store_true', help='Run training function')
    arg.add_argument('--lstm_train', action='store_true', help='Run training function')
    arg.add_argument('--rnn_train', action='store_true', help='Run training function')
    arg.add_argument('--rnn_info', action='store_true', help='Run training function')
    arg.add_argument('--lstm_load', action='store_true', help='Run training function')
    arg.add_argument('--use_trustee', action='store_true', help='Run training function')
    arg.add_argument('--lstm_attack', action='store_true', help='Implement an attack on the LSTM')
    arg.add_argument('--plot', action='store_true')


    args = arg.parse_args()
    if args.upath:
        print('Using saved data')
        df = open_file(filepath)
    elif args.dl:
        print('Downloading data')
        get_data(filepath)
    elif args.lstm_info:
        clear()
        print('LSTM info')
        lstm.summary()
        lg = open_file(filepath)
        lg_X = lg.drop(['EPOCH_diff', 'maneuver_made', 'D_EPOCH'], axis=1)
        lg_y = lg['maneuver_made']
        print(f'Shape X: {lg_X.shape}, Shape y: {lg_y.shape}')
    elif args.rnn_info:
        clear()
        print('RNN info')
        rnn.info()
    elif args.lstm_train:
        clear()
        train(lstm)
    elif args.rnn_train:
        clear()
        train(rnn)
    elif args.lstm_load:
        clear()
        load_mod()
    elif args.use_trustee:
        lstm = load_model('../models/model.keras')
        use_trustee(lstm)
    elif args.lstm_attack:
        attack_model(lstm)
    elif args.plot:
        plot_data(df)

    end = tm.time()
    expr = colored(f'{end-start}', 'green')
    print(f'Total time taken: {expr}')
    
