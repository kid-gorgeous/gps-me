import os 
import httpx
import requests
import subprocess 
import pandas as pd
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from argparse import ArgumentParser

arg = ArgumentParser()

# python api.py wkwargs=[--data --tle --ilrs --train --test --save --load]
arg.add_argument('--data', action='store_true')
arg.add_argument('--tle', action='store_true')
# they call function to control the api 

# Get the environment variables
datapath = os.getenv('DATA_PATH')
user, password = os.environ.get('SPACETRACKER_UNAME'), os.environ.get('SP_PASSWORD')


class N2YO_Client:
    def __init__(self):
        self.api_key = 'E6VD62-BQKN53-EMWXG2-5ECI'
        self.url = 'https://www.n2yo.com/satellites/?c=52&p=A'
        self.client = httpx.Client()

    def download_page(self):
        response = requests.get(self.url)
        with open('data/html/spacex_satellites.html', 'w') as file:
            file.write(response.text)
        print('Page downloaded successfully')
    

# The SpaceTrack API Client that I have created to gather daily TLE data
# the client ables it. Kwargs will be used to control the api [--tle]
class SP_Client:
    def __init__(self, identity, password):
        self.identity = identity
        self.password = password
        self.client = httpx.Client(timeout = 10)
        self.st = SpaceTrackClient(identity, password, httpx_client = self.client)

        self.tle_df_line1 = pd.DataFrame(columns=['Line Number', 'Satellite Number w/ Unclassified', 'Int. Designator', 'Epoch', 'First Time Derivative of Mean Motion', 'Second Time Derivative of Mean Motion', 'BSTAR Drag Term', 'Ephemeris Type', 'Element Number w. Checksum'])
        self.tle_df_line2 = pd.DataFrame(columns=['Line Number', 'Satellite Number w/ Unclassified', 'Inclination', 'Right Ascension of Ascending Node', 'Eccentricity', 'Argument of Perigee', 'Mean Anomaly', 'Mean Motion w/ Revolution Number at Epoch'])
        
    def get(self):
        data = self.st.tle_latest(iter_lines=True, ordinal=1, epoch='>now-30',
                        mean_motion=op.inclusive_range(0.99, 1.01),
                        eccentricity=op.less_than(0.01), format='tle')

        with open('data/tle_latest.txt', 'w') as fp:
            for line in data:
                fp.write(line + '\n')
        print('Data downloaded successfully')

    def get_historicals(self):
        pass

    def save_csv(self, filepath, filename):
        modifiedpath = filepath + '/data.csv'
        filepath = filepath + filename

        df = pd.read_csv(filepath, delimiter="\t")
        df = df.dropna()
        df = df.drop_duplicates()
        df.to_csv(modifiedpath, index=False)
        

    def tle_to_df(self, filepath):
        self.df = pd.read_csv(filepath, delimiter="\t")
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        return self.df



    # ERROR: change file pass 
    def set_data(self):
        with open(f'{datapath}/tle_latest.txt', 'r') as file:
            fileline = file.readlines()
            file_length = len(fileline)
            line = fileline[0]


            # print('File length: ' ,file_length)
            for i in range(file_length):
                if i % 2 == 0:
                    try:
                        line = fileline[i]
                        line = line.split()
                        self.tle_df_line1.loc[i] = line
                    except Exception as e:
                        pass
                else: 
                    try:
                        line = fileline[i]
                        line = line.split()
                        self.tle_df_line2.loc[i] = line
                    except Exception as e:
                        pass
                        
        # Return 
        # TODO: Return the dataframes but with perhaps a classifying column [Maneuver, Non-Maneuver]
             
        
# The ILRS Client that I have created to gather regular ILRS data
# made upon changes it should be able to gather data from the ILRS
# using ftp or http requests to gather satellite data and store it in a dataframe
# using a dataframe generator
class ILRS_Client:
    def __init__(self, satellite): 
        self.record = ['h1', 'h2', 'h3', 'h4', 'h8', 'h9', 'c1', 'c2', 'c3', 'c4', '10', '11', '12', '20', '21', '30','40', '50', '60', '9X', '00']
        self.formatting_header = list()
        self.station_info = list()
        self.spacecraft_info = list()
        self.session_info = list()  

        self.laser_configuration = list()
        self.detector_configuration = list()
        self.timing_configuration = list()
        self.transponder_configuration = list()

        self.range = list()
        self.normal_point = list()
        self.range_supplement = list()

        self.point_angle = list()
        self.calibration = list()
        self.session_statistics = list()
        self.compatiblity = list()

        self.satellite = satellite

        self.h3df = pd.DataFrame(columns=['Record ID', 'Satellite', 'Cospar ID', 'SIC ID', 'Norad ID', 'Epoch Scale', 'Target', 'Location Range'])

    def get(self):
        m = len(open('data/sentinel3a_20220_January.npt', 'r').readlines())
        with open('data/sentinel3a_20220_January.npt', 'r') as file:
            data = file.readlines()
            # span the file line by line
            for i in range(0,1):
                record_id = data[i][0:2]
                line_data = data[i].split()

                # This will process the data from the Sentinel 3A satellite file
                if record_id == self.record[2]:
                    print(line_data)
                    print("Length of line data: ", len(line_data))

                    # COSPAR ID, SIC ID, NORAD ID, EPOCH SCALE, TARGET
                    satellite = line_data[1]
                    cospar_id = line_data[2]
                    sic_id = line_data[3]
                    norad_id = line_data[4]
                    epoch_scale = line_data[5]
                    target = line_data[6]

                    print("Printed r Index: ", record_id,satellite, cospar_id, sic_id, norad_id, epoch_scale, target)
                    r_row = [record_id, satellite,cospar_id, sic_id, norad_id, epoch_scale, target]
                    print("Length of Row: ", len(r_row))
                    print(r_row)
        print((self.h3df.columns))     
        return data


# Test open file function that using the maneuver data
def open_file():
    col = ['Satellite ID', 'Maneuver Year Start', 'Maneuver Day Start', 'Maneuver Hour Start', 'Maneuver Minute Start', 'Maneuver Year End', 'Maneuver Day End', 'Maneuver Hour End', 'Maneuver Minute End', 'Maneuver Type', 'Maneuver Type', 'Number of Burns']
    df = pd.DataFrame(columns=col)
    print("Number of Columns", len(df.columns))
    m = len(open('data/cs2man.csv', 'r').readlines())
    print("Length of file: ", m)
    for i in range(0,1):
        with open('data/cs2man.csv', 'r') as file:
            data = file.readlines()
            line = data[i].split(' ')
            sat_id = line[0]
            
            print("FILE LINE: ", line, "\nNumber of elements: ", len(line))
            print(f"Satellit ID: {sat_id}, Date of Manuever: {line[1:5]}, Date of End: {line[5:9]}")
            
            df.loc[i] = line



if __name__ == "__main__":

    ilrs = ILRS_Client('sentinal3a')
    sp = SP_Client(user, password)  
    args = arg.parse_args()

    if args.data:
        sp.get()
    if args.tle:
        sp.set_data()



