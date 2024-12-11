import os
import json

# Some configuration data
config = {
    # Add configuration data here
    "httpx": {
        "timeout": 10
    },
    "pandas": {
        "display.max_rows": 100
    },
    "spacetrack": {
        # add/ or disregard the env variables
        "identity": f'{os.environ['SPACETRACKER_UNAME']}',
        "password": f'{os.environ['SP_PASSWORD']}',
        # extras env paths
        "datapath": f'{os.environ['DATA_PATH']}',
        "sp_path": f'{os.environ['SAT_PATH']}'
    },
    "ilrs": {
        # add/ or disregard the env variables
        "username": f'{os.environ['ILRS_UNAME']}',
        "password": f'{os.environ['ILRS_PASSWORD']}',
        # extras env paths
        "url": f'{os.environ['ILRS_URL']}' 
        # https://cddis.nasa.gov/archive/slr/data/npt_crd/sentinel3a/2022/
        # implement a URL parser that will define and replace the target files
        ""
    }
    "dev": {
        # please implement this file
    }
}

# Write the configuration data to a JSON file
with open('folder_not_found/config.json', 'w') as f:
    json.dump(config, f, indent=4)


