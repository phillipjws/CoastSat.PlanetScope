# ==========================================================#
# Shoreline Extraction from Satellite Images                #
# ==========================================================#

from coastsat_ps.data_import import initialise_settings
from coastsat_ps.extract_shoreline import extract_shorelines, compute_intersection
from coastsat_ps.interactive import filter_shorelines                    
from coastsat_ps.preprocess import data_extract, pre_process, select_ref_image, add_ref_features
from coastsat_ps.postprocess import ts_plot_single
from coastsat_ps.shoreline_tools import make_animation_mp4, get_point_from_geojson
from coastsat_ps import SDS_slope
from coastsat_ps.plotting import plot_inputs
import pyfes
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pytz
import time
import pickle
import numpy as np
import pandas as pd


def setup_user_input_settings():
    """Define and return user input settings for the processing workflow."""
    settings = {
        'site_name': 'NOBLE_CREEK',
        'cloud_threshold': 15,
        'extent_thresh': 90,
        'output_epsg': '3157',
        'aoi_kml': 'noble_creek.kml',
        'transects': 'NOBLE_CREEK_TRANSECTS.geojson',
        'downloads_folder': r'U:\PS_Imagery\Noble_Creek',
        'classifier': 'NN_8b_new.pkl',
        'im_coreg': 'Local Coreg',
        'generic_land_mask': False,
        'cloud_buffer': 9,
        'max_dist_ref': 150,
        'min_beach_area': 25 * 25,
        'min_length_sl': 100,
        'GDAL_location': r'C:\Users\psteeves\AppData\Local\miniforge3\envs\coastsat_ps\Library\bin',
        'georef_im': False,
        'water_index': 'NmG'
    }
    return settings


def initialise_processing(settings):
    """Initialize the settings for processing and return outputs."""
    outputs = initialise_settings(settings)
    return outputs


def pre_process_data(settings, outputs):
    """Perform pre-processing steps including data extraction, reference image selection, and coregistration."""
    data_extract(settings, outputs)
    select_ref_image(settings, replace_ref_im=False)
    pre_process(settings, outputs, del_files_int=True, rerun_preprocess=True)
    add_ref_features(settings, plot=False, redo_features=True)


def extract_and_filter_shorelines(settings, outputs):
    """Extract shoreline data and filter it for manual corrections."""
    shoreline_data = extract_shorelines(outputs, settings, del_index=True, rerun_shorelines=False, reclassify=False)
    filtered_shoreline_data = filter_shorelines(settings, manual_filter=False, load_csv=False)
    return filtered_shoreline_data


def save_shoreline_animation(settings):
    """Save an animation of the extracted shoreline data."""
    filepath_images = settings['index_png_out']
    fn_out = os.path.join(settings['output_folder'], f"{settings['site_name']}_animation_shorelines.mp4")
    fps = 4
    make_animation_mp4(filepath_images, fps, fn_out)


def compute_transect_intersections(filtered_shoreline_data, settings):
    """Compute intersections along transects and export to CSV."""
    sl_csv = compute_intersection(filtered_shoreline_data, settings)
    return sl_csv


def load_fes_config():
    """Load the FES2022 configuration file."""
    start_time = time.time()
    print("Loading FES2022 config file...")
    config_filepath = os.pardir
    config = os.path.join(config_filepath, 'fes2022.yaml')
    handlers = pyfes.load_config(config)
    runtime = time.time() - start_time
    print(f"Config file loaded in {int(runtime)} seconds, ({str(round(runtime/60, 2))} minutes)")
    return handlers['tide'], handlers['radial']

def get_tide_data(settings, sl_csv, ocean_tide, load_tide):
    centroid = get_point_from_geojson(settings)
    centroid[0] = centroid[0] + 360 if centroid[0] < 0 else centroid[0]
    print(f"Adjusted centroid: {centroid}")

    date_range = [
        pytz.utc.localize(datetime(2014, 1, 1)),
        pytz.utc.localize(datetime(2025, 1, 1)),
    ]
    timestep = 900

    dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, timestep, ocean_tide, load_tide)
    dates_sat = sl_csv['Date'].tolist()
    tides_sat = SDS_slope.compute_tide_dates(centroid, dates_sat, ocean_tide, load_tide)

    return dates_sat, dates_ts, tides_sat, tides_ts

def estimate_beach_slope(settings, sl_csv, dates_sat, dates_ts, tides_sat, tides_ts):
    fp_slopes = settings['beach_slope_out']
    print(f'Loading pickle file: {settings["sl_pkl_file"]}')
    
    with open(settings['sl_pkl_file'], 'rb') as f:
        output = pickle.load(f)

    if isinstance(output, dict):
        print("Keys in the pickle file:", list(output.keys()))
    else:
        print("Loaded data is not a dictionary.")

    # Prepare settings
    settings_slope = {
        'slope_min': 0.005,
        'slope_max': 0.6,
        'delta_slope': 0.005,
        'n0': 50,
        'freq_cutoff': 1. / (24 * 3600 * 30),  # 30 day frequency
        'delta_f': 100 * 1e-10,
        'prc_conf': 0.05,
        'plot_fig': True,
    }

    beach_slopes = SDS_slope.range_slopes(settings_slope['slope_min'], settings_slope['slope_max'], settings_slope['delta_slope'])
    beach_slopes = np.clip(beach_slopes, None, 0.6)
    settings_slope['date_range'] = [2020, 2024]

    # Convert date range to datetime objects
    settings_slope['date_range'] = [pytz.utc.localize(datetime(settings_slope['date_range'][0], 8, 1)),
                                    pytz.utc.localize(datetime(settings_slope['date_range'][1], 7, 1))]

    # Ensure dates_sat are datetime objects
    dates_sat = [pytz.utc.localize(datetime.strptime(date_str, '%Y-%m-%d')) if isinstance(date_str, str) else date_str for date_str in dates_sat]

    dates_sat = pd.to_datetime(sl_csv['Date'], utc=True).tolist()
    tides_sat = np.array(tides_sat)

    # Filter dates_sat and tides_sat based on date_range
    idx_dates = [settings_slope['date_range'][0] < date < settings_slope['date_range'][1] for date in dates_sat]
    selected_indices = np.where(idx_dates)[0]

    # Apply filtering
    dates_sat = [dates_sat[i] for i in selected_indices]
    tides_sat = np.array(tides_sat)[selected_indices]

    # Apply filtering on sl_csv and ensure lengths match
    sl_csv = sl_csv.iloc[selected_indices].reset_index(drop=True)

    SDS_slope.plot_timestep(dates_sat)
    plt.gcf().savefig(os.path.join(fp_slopes, '0_timestep_distribution.jpg'), dpi=200)

    settings_slope['n_days'] = 1

    settings_slope['freqs_max'] = SDS_slope.find_tide_peak(dates_sat, tides_sat, settings_slope)
    plt.gcf().savefig(os.path.join(fp_slopes, '1_tides_power_spectrum.jpg'), dpi=200)

    slope_est, cis = dict(), dict()
    transect_columns = settings['transects_load'].keys()

    for key in transect_columns:
        idx_nan = np.isnan(sl_csv[key])
        dates = [dates_sat[i] for i in np.where(~idx_nan)[0]]
        tide = tides_sat[~idx_nan]
        composite = sl_csv[key][~idx_nan].to_numpy(dtype=np.float64)

        tsall = SDS_slope.tide_correct(composite, tide, beach_slopes)
        slope_est[key], cis[key] = SDS_slope.integrate_power_spectrum(dates, tsall, settings_slope, key)
        
        plt.gcf().savefig(os.path.join(fp_slopes, f'2_energy_curve_{key}.jpg'), dpi=200)
        SDS_slope.plot_spectrum_all(dates, composite, tsall, settings_slope, slope_est[key])
        plt.gcf().savefig(os.path.join(fp_slopes, f'3_slope_spectrum_{key}.jpg'), dpi=200)
        
        print(f'Beach slope at transect {key}: {slope_est[key]:.3f} ({cis[key][0]:.4f} - {cis[key][1]:.4f})')

    return slope_est


def tidal_correction(settings, tide_settings, sl_csv, dates_sat, dates_ts, tides_sat, tides_ts):
    """Perform tidal correction on shoreline positions and save results."""

    reference_elevation = tide_settings['contour']
    beach_slope = tide_settings['beach_slope']

    for transect in settings['transects_load'].keys():
        if type(tide_settings['beach_slope']) is not list:
            beach_slope = tide_settings['beach_slope']
        else:
            beach_slope = tide_settings['beach_slope'][transect]
        correction = (tides_sat - reference_elevation) / beach_slope[transect]
        sl_csv[transect] += correction

    if tide_settings.get('save_figure', False):
        fig, ax = plt.subplots(figsize=(15, 4), tight_layout=True)
        ax.grid(which='major', linestyle=':', color='0.5')
        ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
        ax.plot(dates_sat, tides_sat, '-o', color='k', ms=6, mfc='w', lw=1, label='image acquisition')
        ax.set(ylabel='Tide level [m]', xlim=[dates_sat[0], dates_sat[-1]], title='Tide levels at the time of image acquisition')
        ax.legend()
        fig.savefig(os.path.join(settings['output_folder'], 'tide_timeseries.jpg'), dpi=200)

    sl_csv.to_csv(os.path.join(settings['output_folder'], 'transect_time_series_tidally_corrected.csv'), index=False)
    print('Tidally corrected shoreline data saved.')
    return sl_csv


def plot_transects(settings, sl_csv_tide):
    """Plot time-series of shoreline positions along transects."""
    for transect in settings['transects_load'].keys():
        ts_plot_single(settings, sl_csv_tide, transect, savgol=True, x_scale='years')


def main():
    """Main function to execute the entire shoreline extraction and analysis workflow."""
    settings = setup_user_input_settings()
    outputs = initialise_processing(settings)
    pre_process_data(settings, outputs)
    
    shoreline_data = extract_and_filter_shorelines(settings, outputs)
    sl_csv = compute_transect_intersections(shoreline_data, settings)

    ocean_tide, load_tide = load_fes_config()

    dates_sat, dates_ts, tides_sat, tides_ts = get_tide_data(settings, sl_csv, ocean_tide, load_tide)

    tide_settings = {
        'beach_slope': 0.1,
        'contour': 0.0,
        'save_figure': True,
        'date_min': '2014-01-01',
        'date_max': '2025-01-01'
    }
    tide_settings['beach_slope'] = estimate_beach_slope(settings, sl_csv, dates_sat, dates_ts, tides_sat, tides_ts)
    sl_csv_tide = tidal_correction(settings, tide_settings, sl_csv, dates_sat, dates_ts, tides_sat, tides_ts)
    plot_transects(settings, sl_csv_tide)
    # plt.show()


if __name__ == '__main__':
    main()
