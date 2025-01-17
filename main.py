from spacetrack import SpaceTrackClient
from ipregistry import IpregistryClient
from argparse import ArgumentParser
import spacetrack.operators as op
from geopy.geocoders import Nominatim
import pandas as pd
import requests
import geocoder
from datetime import datetime
import datetime as dt
import json as js
import threading
import httpx
import time
import sys
import os


# Credentials to access the Space-track API
identity = os.environ.get('SPACETRACKER_UNAME')
password = os.environ.get('SP_PASSWORD')
expr = f'{identity}'
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


def get_location_by_ip():
    response = requests.get('https://ipinfo.io')
    data = response.json()
    location = data['loc'].split(',')
    latitude, longitude = location[0], location[1]
    return latitude, longitude

def get_location_by_geocoder():
    g = geocoder.ip('me')
    return g.latlng


def test_location():
    latitude, longitude = get_location_by_ip()
    print("Latitude = ", latitude)
    print("Longitude = ", longitude)
    from ipregistry import IpregistryClient


if __name__ == '__main__':

    arg = ArgumentParser()
    arg.add_argument('--location', action='store_true')
    args = arg.parse_args()
    if args.location:
        print(get_location_by_geocoder())


