import os 
import httpx
import subprocess 
import pandas as pd
from spacetrack import SpaceTrackClient
import spacetrack.operators as op


class SP_Client:
    def __init__(self, identity, password):
        self.identity = identity
        self.password = password
        self.client = httpx.Client(timeout = 10)
        self.st = SpaceTrackClient(identity, password, httpx_client = self.client)
      
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
        df = pd.read_csv(filepath, delimiter="\t")
        df = df.dropna()
        df = df.drop_duplicates()
        return df

    # def get_training_data(self):
    #     # query = self.st.tle_latest(iter_lines=True, ordinal=1, epoch='>now-30',
    #     #                 mean_motion=op.inclusive_range(0.99, 1.01),
    #     #                 eccentricity=op.less_than(0.01), format='tle')

    #     tle_data = self.st.tle()
    #     tle_df = pd.DataFrame(tle_data)

    #     # train, and test
    #     print(tle_df.iloc[:80], tle_df.iloc[80:])
    #     # return tle_df.iloc[:80], tle_df.iloc[80:]


class ILRS_Client:
    def __init__(self):
        self.record = ['h1', 'h2', 'h3', 'h4', 'h8', 'h9', 'c1', 'c2', 'c3', 'c4', '10', '11', '12', '20', '21', '30','40', '50', '60', '9X', '00']
        # header h1
        self.formatting_header = list()
        # header h2
        self.station_info = list()
        # header h3
        self.spacecraft_info = list()
        # header h4
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

        self.satellite = str()

        pass

    def get(self):
        m = len(open('data/sentinel3a_20220_January.npt', 'r').readlines())
        with open('data/sentinel3a_20220_January.npt', 'r') as file:
            data = file.readlines()
            # span the file line by line
            for i in range(0,m):
                record_id = data[i][0:2]
                line_data = data[i].split()



                # 
                if record_id == self.record[2]:
                    self.spacecraft_info.append(line_data)
                    # print(line_data)
                    self.satellite = line_data[1]
                    self.norad_id = line_data[3]
                    print('Satellite: ', self.satellite, 'Norad ID: ', self.norad_id)   


        return data

        
sp = SP_Client(os.environ.get('SPACETRACKER_UNAME'), os.environ.get('SP_PASSWORD'))          
tle_df = sp.tle_to_df('/Users/evan/Documents/School/Fall2023_Spring2024/Cyber/finalproject/data/tle_latest.txt')                
tle_df = tle_df.drop(0)
for line in tle_df:
    print(type(line))
    print(line)




ilrs = ILRS_Client()
# slr_data = ilrs.get()

