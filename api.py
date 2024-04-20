import os 
import httpx
import subprocess 
import pandas as pd
import spacetrack.operators as op
from spacetrack import SpaceTrackClient
from argparse import ArgumentParser


arg = ArgumentParser()
arg.add_argument('--data', action='store_true', help='')
arg.add_argument('--tle', action='store_true', help='')
arg.add_argument('--ilrs', action='store_true', help='')
arg.add_argument('--train', action='store_true', help='')
arg.add_argument('--test', action='store_true', help='')
arg.add_argument('--save', action='store_true', help='')
arg.add_argument('--load', action='store_true', help='')


datapath = os.getenv('DATA_PATH')
user, password = os.environ.get('SPACETRACKER_UNAME'), os.environ.get('SP_PASSWORD')


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

    def set_data(self):
        with open('/Users/evan/Documents/School/Fall2023_Spring2024/Cyber/finalproject/data/tle_latest.txt', 'r') as file:
            fileline = file.readlines()
            file_length = len(fileline)
            line = fileline[0]

            print('File length: ' ,file_length)
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
                        
    def train(self):
        pass
        


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

        self.df = pd.DataFrame(columns=['Record ID', 'Satellite', 'Cospar ID', 'Norad ID', 'Year', 'Month', 'Day', 'Hour', 'Minute', 'Second', 'Millisecond', 'Microsecond', 'UTC', 'Range', 'Normal Point', 'Range Supplement', 'Point Angle', 'Calibration', 'Session Statistics', 'Compatibility'])

    def get(self):
        m = len(open('data/sentinel3a_20220_January.npt', 'r').readlines())
        with open('data/sentinel3a_20220_January.npt', 'r') as file:
            data = file.readlines()
            # span the file line by line
            for i in range(0,m):
                record_id = data[i][0:2]
                line_data = data[i].split()

                # This will process the data from the Sentinel 3A satellite file
                if record_id == self.record[2]:
                    print(line_data)
                    # Append the line data to the spacecraft_info list
                    self.spacecraft_info.append(line_data)
                    self.satellite = line_data[1]
                    self.cospar_id = line_data[2]
                    self.norad_id = line_data[4]


                    # print('Satellite: ', self.satellite, 'Norad ID: ', self.norad_id)   
                
        return data



if __name__ == "__main__":

    ilrs = ILRS_Client('sentinal3a')
    sp = SP_Client(user, password)  
    args = arg.parse_args()

    if args.data:
        sp.get()
    if args.tle:
        sp.set_data()
        # print(sp.tle_df_line1)
    if args.ilrs:
        ilrs.get()
    if args.test:
        pass
    if args.save:
        sp.save_csv()
    if args.load:
        pass


