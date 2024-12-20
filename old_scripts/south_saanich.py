#%% Imports
from coastsat_ps.data_import import initialise_settings
from coastsat_ps.extract_shoreline import extract_shorelines, compute_intersection
from coastsat_ps.interactive import filter_shorelines                    
from coastsat_ps.preprocess import (data_extract, pre_process, select_ref_image, 
                                    add_ref_features)
from coastsat_ps.postprocess import ts_plot_single
from coastsat_ps import SDS_slope
import pyfes
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import copy

#%% User Input Settings
settings = {
    ### General Settings ###
    'site_name': 'SOUTH_SAANICH',
    'cloud_threshold': 2,
    'extent_thresh': 95,
    'output_epsg': '3157',

    ### Reference files ###
    'aoi_kml': 'south_saanich.kml',
    'transects': False, 
    'downloads_folder': r'U:\PS_Imagery\South_Saanich_psscene_analytic_8b_udm2\PSScene',

    ### Processing settings ###
    'classifier': 'NN_8b_new.pkl',
    'im_coreg': 'Local Coreg',
    'generic_land_mask': False,

    ### Advanced settings ###
    'cloud_buffer': 9, 
    'max_dist_ref': 150,
    'min_beach_area': 50*50, 
    'min_length_sl': 500,
    'GDAL_location': r'C:\Users\psteeves\AppData\Local\miniforge3\envs\coastsat_ps\Library\bin',
    
    'water_index': 'NmB'
}

outputs = initialise_settings(settings)

#%% Pre-processing - TOA conversion and mask extraction
data_extract(settings, outputs)

#%% Select reference image for coreg
select_ref_image(settings, replace_ref_im=True)

#%% Image coreg and scene merging, run manually
pre_process(settings, outputs, del_files_int=True, rerun_preprocess=False)

#%% Add reference Features
add_ref_features(settings, plot=False, redo_features=False)

#%% Extract shoreline data
shoreline_data = extract_shorelines(outputs, settings, del_index=False, rerun_shorelines=True, reclassify=True)

#%% Manual Error Detection
shoreline_data = filter_shorelines(settings, manual_filter=True, load_csv=False)

#%% Transect intersection and csv export
sl_csv = compute_intersection(shoreline_data, settings)

#%% Load Pyfes
print("Loading FES2022 config file...")
# Load FES2022 configuration
config_filepath = os.pardir
config = os.path.join(config_filepath, 'fes2022.yaml')
handlers = pyfes.load_config(config)
print("Config file loaded")
ocean_tide = handlers['tide']
load_tide = handlers['radial']

#%% Tidal Correction
def tidal_correction(settings, tide_settings, sl_csv):
    """
    Perform tidal correction on the shoreline positions using pyfes.
    """
    # Compute the centroid of the polygon
    centroid = [-123.47743715625634, 48.58973432186312]
    centroid[0] = centroid[0] + 360 if centroid[0] < 0 else centroid[0]
    print(f"Adjusted centroid: {centroid}")

    # Set date range and timestep for tidal calculations
    date_range = [
        pytz.utc.localize(datetime(2014, 1, 1)),
        pytz.utc.localize(datetime(2025, 1, 1)),
    ]
    timestep = 900  # seconds (15 minutes)

    # Compute tide levels for the entire time range
    dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, timestep, ocean_tide, load_tide)

    # Get tide levels for image acquisition dates
    dates_sat = sl_csv['Date'].tolist()
    tides_sat = SDS_slope.compute_tide_dates(centroid, dates_sat, ocean_tide, load_tide)

    # Tidal correction parameters
    reference_elevation = tide_settings['contour']
    beach_slope = tide_settings['beach_slope']

    # Apply tidal correction
    for transect in settings['transects_load'].keys():
        correction = (tides_sat - reference_elevation) / beach_slope
        sl_csv[transect] += correction

    # Optionally, save a figure showing the tide levels
    if tide_settings.get('save_figure', False):
        fig, ax = plt.subplots(1, 1, figsize=(15, 4), tight_layout=True)
        ax.grid(which='major', linestyle=':', color='0.5')
        ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
        ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w', lw=1, label='image acquisition')
        ax.set(ylabel='Tide level [m]', xlim=[dates_sat[0], dates_sat[-1]], title='Tide levels at the time of image acquisition')
        ax.legend()
        fig.savefig(os.path.join(settings['output_folder'], 'tide_timeseries.jpg'), dpi=200)

    # Save the tidally corrected CSV file
    sl_csv.to_csv(os.path.join(settings['output_folder'], 'transect_time_series_tidally_corrected.csv'), index=False)
    print('Tidally corrected shoreline data saved.')

    return sl_csv

#%% Define tidal correction settings
tide_settings = {
    'beach_slope': 0.1,  # Example beach slope value
    'contour': 0.0,      # Reference elevation contour
    'save_figure': True, # Set to True to save figure
    'date_min': '2014-01-01',
    'date_max': '2025-01-01'
}

# Apply tidal correction
sl_csv_tide = tidal_correction(settings, tide_settings, sl_csv)

#%% Plot transects
for transect in settings['transects_load'].keys():
    ts_plot_single(settings, sl_csv_tide, transect,
                   savgol=True,
                   x_scale='years')
