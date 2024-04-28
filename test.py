'''
This script downloads data from space-track.org. 
'''
def clear():
    import os
    os.system('clear')


from termcolor import colored
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Import spacetrack library modules
from spacetrack import SpaceTrackClient
import spacetrack.operators as op

# Import auxiliary libraries
import datetime as dt
import pandas as pd
import numpy as np
import time as tm
import json as js
import threading
import httpx
import sys

import matplotlib.pyplot as plt
from datetime import datetime

import os

# Credentials to access the Space-track API
identity = os.environ.get('SPACETRACKER_UNAME')
password = os.environ.get('SP_PASSWORD')

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

def print_runtime(start, end, function_id):
    print(f'Time taken: {end-start} for function class {function_id}')

def parse_data(id, omm_data, filepath):
    function_id = 'parse_data'

    # Parse the data and save into a dataframe
    omm = js.loads(omm_data)
    df  = pd.DataFrame(omm)
    fileName = filepath + f'/sat-data{id}.csv'
    df.to_csv(fileName, mode='a', index=False)
    print(f"Data saved to {fileName}")
    print(f'Columns: {df.columns}')

def get_data(filepath):
    time = tm.time()
    function_id = 'get_data'
    print(f'Logging in with {identity}')

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
        omm = st.omm( norad_cat_id=id_list , format='json', epoch=drange)
        parse_data(id, omm, filepath)

        print("Completed request")



        # Save the current id
        # current_id = id+win_size
        # with open("last_completed_initial_{id}.csv", "w") as file:
        #     file.write(str(current_id))



    end = tm.time()
    print(f'Time taken: {end-time} for function class {function_id}')

def plot_data(df, df_diff):
    function_id = 'plot_data'
    print(f'Plotting the data...')
    df.plot()
    plt.show()

# open the file and read the data from the file path
def open_file(filepath):
    start = tm.time()
    function_id = 'open_file'
    for i in range(1):
        filename = filepath + f'/sat-data{ids[i]}.csv'
        term = colored(f'Opening {filename}', 'green')
        clear(), print(f'{term}')
        data = pd.read_csv(filename, dtype=str, delimiter=',')

        le = LabelEncoder()
        print('Columns: ')
        # EPOCK: 10, 
        print(data.columns[10])
        EPOCH = data.columns[10]
        # Inclination: 13
        print(data.columns[13])
        INCLINATION = data.columns[13]
        # RAAN: 14
        print(data.columns[14])
        RAAN = data.columns[14]
        # Mean anomaly: 16
        print(data.columns[16])
        MEAN_ANOMALY = data.columns[16]
        # Semi-major axis: 28
        print(data.columns[28])
        SEMIMAJOR_AXIS = data.columns[28]
    

        data_cols = [EPOCH, INCLINATION, RAAN, MEAN_ANOMALY, SEMIMAJOR_AXIS]
        df = pd.DataFrame(data, columns=data_cols)
        df['EPOCH'] = pd.to_datetime(df['EPOCH'], format='mixed', errors='coerce')
        df['EPOCH'] = pd.to_datetime(df['EPOCH'], unit='s')
        df['EPOCH'] = df['EPOCH'].astype(int) / 10**9
        df['EPOCH'] = df['EPOCH'].astype('datetime64[s]').astype(int)
        
        print('\n',df.shape)
        print('\n',df.head())


        # Calculate the difference between the columns
        df_diff = df.copy()
        df_diff['EPOCH'] = pd.to_datetime(df_diff['EPOCH'], unit='s')
        df_diff['EPOCH'] = df_diff['EPOCH'].astype(int) / 10**9
        df_diff['EPOCH'] = df_diff['EPOCH'].astype('datetime64[s]').astype(int)
        df_diff.columns = ['D_' + col for col in data_cols]
        print('\n',df_diff.shape)
        print('\n',df_diff.head())

        # Concatenate df and df_diff
        df_combined = pd.concat([df, df_diff], axis=1)
        df_combined = df_combined.dropna()
        df_combined = df_combined.reset_index(drop=True)
        df_combined['EPOCH'] = pd.to_datetime(df_combined['EPOCH'], unit='s')
        df_combined['EPOCH'] = df_combined['EPOCH'].astype(int) / 10**9
        df_combined['EPOCH'] = df_combined['EPOCH'].astype('datetime64[s]').astype(int)
        df_combined['EPOCH_diff'] = df_combined['EPOCH'].diff()
        
        # df_combined['EPOCH_diff'] = le.fit_transform(df_combined['EPOCH_diff'])
        # df_combined['EPOCH_diff'] = df_combined['EPOCH_diff'].astype('category').cat.codes
        # onehot_df_combined = pd.get_dummies(df_combined, columns=['EPOCH_diff'])
        
        # Define the class boundaries
        bins = pd.to_timedelta(['-1 days', '10 days', '100 days', '1000 days'])
        labels = ['small', 'medium', 'large']

        # Classify EPOCH_diff
        df_combined['EPOCH_diff_class'] = pd.cut(df_combined['EPOCH_diff'], bins=bins, labels=labels)
        df_combined = df_combined.drop('EPOCH_diff', axis=1)
        df_combined = df_combined.dropna()

        print('\n', df_combined.shape)
        # print('\n', df_combined.head())
        print('\n')

        print_runtime(start, tm.time(), function_id)
        return df_combined

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

def _preprocess(get_request):
    function_id = 'preprocess'
    print("Start preprocessing data... create index, drop duplicates, and localize feature labels")
    open_file(get_request)
    # print(f'Datatype of property: {type(get_request)}')
    # print(f'Length of property: {len(get_request)}')




# Main function w/ argparse requestType and Exception handling

def main(filepath, requestType):
    data_get_start = tm.time()
    try:
        if requestType == '--rpath':
            print('Getting data...')
            get_data(filepath)
            print('Using filepath: ', filepath)
        else: # using --upath
            _preprocess(filepath)
    except Exception as e:
        expr = colored('Exception in main(): ', 'green')
        print(f'{expr}', str(e))

    data_get_end = tm.time()
    print_runtime(data_get_start, data_get_end, 'get_data')


# Start the timer and run the main function
start  = tm.time()
if __name__ == '__main__':
    filepath = '/Users/evan/Documents/School/Fall2023_Spring2024/Cyber/finalproject/data/sat-data'
    from argparse import ArgumentParser

    from lstm import K_LSTM
    lstm = K_LSTM(64, 16, 1)

    arg = ArgumentParser()
    arg.add_argument('--upath',action='store_true', help='Using saved data')
    arg.add_argument('--rpath', action='store_true',help='Require data')
    arg.add_argument('--lstm_info', action='store_true', help='Run training function')
    arg.add_argument('--train', action='store_true', help='Run training function')

    args = arg.parse_args()
    if args.upath:
        print('Using saved data')
        main(filepath, args.upath)
    elif args.rpath:
        print('Require data')
        main(filepath, args.rpath)
    elif args.lstm_info:
        clear()
        print('LSTM info')
        lstm.summary()
    elif args.train:
        clear()
        try:
            print('Training the model...')
            df_combined = open_file(filepath)
            print('Dataframe Type:  ', type(df_combined))

            # Assuming df_combined is your DataFrame and 'target' is your class label
            X = df_combined.drop('EPOCH_diff_class', axis=1)
            X_epoch = df_combined['EPOCH']
            print(X_epoch)
            y = df_combined['EPOCH_diff_class']

            y_coded = pd.get_dummies(y)
            for i in range(len(X_epoch)):
                X_epoch[i] = X_epoch[i].timestamp()
                y_coded[i] = X_epoch[i].timestamp() - y[i].timestamp()
                print(X_epoch[i], y_coded[i])


            print("The shape of X,y: ",X.shape, y.shape)

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Reshape the data for the LSTM model
            # X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
            # X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
            # lstm.train(X_train, y_train)
            # lstm.save('/Users/evan/Documents/School/Fall2023_Spring2024/Cyber/finalproject/models/lstm_model.h5')
            print('Model trained successfully')
            
        except Exception as e:
            expr = colored('Exception in main(): ', 'red')
            print(f'{expr}', str(e))
    else:
        print('No arguments passed')

    end = tm.time()
    print(f'Total time taken: {end-start}')
