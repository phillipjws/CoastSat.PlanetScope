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


def setup_user_input_settings():
    """Define and return user input settings for the processing workflow."""
    settings = {
        'site_name': 'UPPER_SIDNEY_ISLAND',
        'cloud_threshold': 5,
        'extent_thresh': 95,
        'output_epsg': '3157',
        'aoi_kml': 'upper_sidney_island.kml',
        'transects': False,
        'downloads_folder': r'U:\PS_Imagery\Upper_Sidney_Island',
        'classifier': 'NN_8b_new.pkl',
        'im_coreg': 'Global Coreg',
        'generic_land_mask': False,
        'cloud_buffer': 9,
        'max_dist_ref': 150,
        'min_beach_area': 50 * 50,
        'min_length_sl': 400,
        'GDAL_location': r'C:\Users\psteeves\AppData\Local\miniforge3\envs\coastsat_ps\Library\bin',
        'georef_im': False,
        'water_index': 'NmB'
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
    pre_process(settings, outputs, del_files_int=True, rerun_preprocess=False)
    add_ref_features(settings, plot=False, redo_features=True)


def extract_and_filter_shorelines(settings, outputs):
    """Extract shoreline data and filter it for manual corrections."""
    shoreline_data = extract_shorelines(outputs, settings, del_index=False, rerun_shorelines=True, reclassify=True)
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
    print("Loading FES2022 config file...")
    config_filepath = os.pardir
    config = os.path.join(config_filepath, 'fes2022.yaml')
    handlers = pyfes.load_config(config)
    print("Config file loaded")
    return handlers['tide'], handlers['radial']


def tidal_correction(settings, tide_settings, sl_csv, ocean_tide, load_tide):
    """Perform tidal correction on shoreline positions and save results."""
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

    reference_elevation = tide_settings['contour']
    beach_slope = tide_settings['beach_slope']

    for transect in settings['transects_load'].keys():
        correction = (tides_sat - reference_elevation) / beach_slope
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
    tide_settings = {
        'beach_slope': 0.1,
        'contour': 0.0,
        'save_figure': True,
        'date_min': '2014-01-01',
        'date_max': '2025-01-01'
    }
    
    sl_csv_tide = tidal_correction(settings, tide_settings, sl_csv, ocean_tide, load_tide)
    plot_transects(settings, sl_csv_tide)
    plt.show()


if __name__ == '__main__':
    main()
