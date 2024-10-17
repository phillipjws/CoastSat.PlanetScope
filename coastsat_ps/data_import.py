# Initialisation and Data import functions
# YD, Sep 2020

import os
import datetime
import numpy as np
from osgeo import gdal
from sklearn.externals import joblib
import json
from shapely.geometry import shape
from shapely.ops import transform
from pyproj import Transformer

from coastsat_ps.preprocess_tools import create_folder
from coastsat_ps.interactive import transects_from_geojson


#%% Pre-Processing Functions

def initialise_settings(settings):
    ''' Initialise user inputs and output folders '''
    
    
    ############# Additional advanced settings ###############################
        
    
    ### Coregistration
    # Georectified TOA reference image path (in CoastSat.PlanetScope/user_inputs folder)
    settings['georef_im'] = False #'Narrabeen_im_ref.tif', # Possible to provide a reference image filepath for coregistration instead of manually selecting from a popup window
    # Tie point averaged x/y local shift for coregistration
    settings['local_avg'] = True
    # Tie-point grid spacing (pixels)
    settings['grid_size'] = 50
    # Tie-point comparison window size (pixels)
    settings['window_size'] = (100,100)
    # Error filtering level (0-3: lower number less likely to fail but less accurate)
    settings['filter_level'] = 2 #3 often fails, bug?
    # Workaround for arosics inability to coregister images with different CRS. Reprojects all TOA/mask files to output epsg first. 
    settings['arosics_reproject'] = False
    # GDAL warp CRS re-sampling method. 
        # 'near' is the fastest/default but images may be jagged as no is smoothing applied. 'cubic' & 'cubicspline' look the best but are slowest. 'bilinear' is a good middle ground. 
        # Note that re-sampling using cubic, cubicspline and bilinear options may cause issues with arosics. 
    settings['gdal_method'] = 'near'
    # Land mask cleaning smoothing parameters - choose lower values if land mask does not cover thin land regions (ie small barrier islands)
    settings['land_mask_smoothing_1'] = 15 # pixels (so x3 for metres)
    settings['land_mask_smoothing_2'] = 10 # pixels (so x3 for metres)
    
    ### Shoreline extraction method
    # Water index
    settings['water_index'] = 'NDWI'
    # Shoreline thresholding method ['Otsu', 'Peak Fraction']
    settings['thresholding'] = 'Peak Fraction'
    # Fraction used in custom peak fraction thresholding method
    settings['peak_fraction'] = 0.7
    # Number of bins for threshold histogram plot
    settings['otsu_hist_bins'] = 150
    # Include PS determined faulty pixels in cloud mask
    settings['faulty_pixels'] = True

    
    ### Thin beach width fixes
    # Generic detection region - if False individual masks are extracted (slower but more accurate sl extraction)
    settings['generic_sl_region'] = False   # Use True for beach w no sand when having issues with classifier
                                            # When true, shoreline is based on a generic crop and otsu and does not use a classified image 
    # Tweak to sl mask for thin/non-existant beach width
    settings['thin_beach_fix'] = True


    ### Transect intersection settings
    # Search width adjacent to transect for SL intersections [in metres]
    settings['along_dist'] = 25
    # Group statistics for filtering intersection points in transect search area [in metres]
    settings['max_std'] = 10
    settings['max_range'] = 25
    settings['min_val'] = 0
    settings['min no. intercepts'] = 5 
    
    
    ##########################################################################
    
    # Check filepath is acceptible for GDAL
    if ' ' in os.getcwd():
        raise Exception('Ensure no whitespace in filepath to CoastSat.PlanetScope folder as this causes a gdal error. Edit filepath or move run folder to a new location.')
    
    # Ensure GDAL directory exists
    if os.path.isdir(settings['GDAL_location']) == False:
        raise Exception('Ensure GDAL location entered is correct')
        
    # Ensure working directory is coastsat
    if not (os.getcwd()[-20:] != 'planetscope_coastsat' or
            os.getcwd()[-25:] != 'CoastSat.PlanetScope-main'):
        raise Exception('Change working directory to CoastSat.PlanetScope or CoastSat.PlanetScope-main.' +
                        'This line can be commented out if a different folder name is wanted. Check is here to ensure working directory is deliberate so data is not saved in a random location ')
    
    # Create output_folders
    settings['outputs_base_folder'] = create_folder(os.path.join(os.getcwd(),'outputs'))
    settings['output_folder'] = create_folder(os.path.join(settings['outputs_base_folder'],settings['site_name']))  # Run directory
    
    settings['toa_out'] = create_folder(os.path.join(settings['output_folder'],'toa_image_data'))
    settings['raw_data'] = create_folder(os.path.join(settings['toa_out'],'raw_data'))
    settings['merge_out_base'] = create_folder(os.path.join(settings['toa_out'],'merged_data'))
    if settings['im_coreg'] == 'Local Coreg':
        settings['merge_out'] = create_folder(os.path.join(settings['merge_out_base'],'local_coreg_merged'))
    elif settings['im_coreg'] == 'Global Coreg':
        settings['merge_out'] = create_folder(os.path.join(settings['merge_out_base'],'global_coreg_merged'))
    elif settings['im_coreg'] == 'Coreg Off':
        settings['merge_out'] = create_folder(os.path.join(settings['merge_out_base'], settings['im_coreg']))
    else:
        raise Exception('Check co-registration setting selection spelling')

    settings['tif_out'] = create_folder(os.path.join(settings['output_folder'],'index tif outputs'))
    settings['index_tif_coreg'] = create_folder(os.path.join(settings['tif_out'], settings['im_coreg']))
    settings['index_tif_out'] = create_folder(os.path.join(settings['index_tif_coreg'], settings['water_index']))
    
    settings['sl_png'] = create_folder(os.path.join(settings['output_folder'],'shoreline outputs'))
    settings['sl_png_coreg'] = create_folder(os.path.join(settings['sl_png'], settings['im_coreg']))
    settings['sl_threshold'] = create_folder(os.path.join(settings['sl_png_coreg'], settings['water_index']))
    settings['sl_thresh_ind'] = create_folder(os.path.join(settings['sl_threshold'], settings['thresholding']))      # shoreline data out folder
    settings['index_png_out'] = create_folder(os.path.join(settings['sl_thresh_ind'], 'Shoreline plots'))

    
    # Create filepaths
    settings['user_input_folder'] = os.path.join(os.getcwd(), 'user_inputs')
    settings['run_input_folder'] = create_folder(os.path.join(settings['output_folder'], 'input_data'))
    
    settings['sl_pkl_file'] = os.path.join(settings['sl_thresh_ind'], settings['site_name'] + '_' + settings['water_index'] + '_' + settings['thresholding'] + '_shorelines.pkl')      # Results out
    settings['sl_geojson_file'] = os.path.join(settings['sl_thresh_ind'], settings['site_name'] + '_' + settings['water_index'] + '_' + settings['thresholding'] + '_shorelines.geojson')
    settings['sl_transect_csv'] = os.path.join(settings['sl_thresh_ind'], settings['site_name'] + '_' + settings['water_index'] + '_' + settings['thresholding'] + '_transect_SL_data.csv')

 
    # Initialise settings
    settings['output_epsg'] = 'EPSG:' + settings['output_epsg']
    settings['pixel_size'] = 3
    settings['min_beach_area_pixels'] = np.ceil(settings['min_beach_area']/settings['pixel_size']**2)
    settings['ref_merge_im_txt'] = os.path.join(settings['run_input_folder'], settings['site_name'] + '_TOA_path.txt')


    # Initialise classifiers [Could use seperate classifiers for coreg and sl extraction?]
    class_path = os.path.join(os.getcwd(),'coastsat_ps', 'classifier', 'models', settings['classifier'])
    land_class_path = os.path.join(os.getcwd(),'coastsat_ps', 'classifier', 'models', settings['classifier'])
    settings['classifier_load'] = joblib.load(class_path)
    settings['land_classifier_load'] = joblib.load(land_class_path)


    # Import transects
    if settings['transects'] != False:
        settings['geojson_file'] = os.path.join(settings['user_input_folder'], settings['transects'])
        settings['transects_load'] = transects_from_geojson(settings['geojson_file'])


    # Update coreg settings 
    if settings['im_coreg'] != 'Coreg Off':
        settings['land_mask'] = os.path.join(settings['run_input_folder'], settings['site_name'] + '_land_mask.tif')
        if settings['im_coreg'] == 'Local Coreg':
            settings['coreg_out'] = create_folder(os.path.join(settings['toa_out'],'local_coreg_data'))
        if settings['im_coreg'] == 'Global Coreg':
            settings['coreg_out'] = create_folder(os.path.join(settings['toa_out'],'global_coreg_data'))

        if settings['georef_im'] != False:
            settings['georef_im_path'] = os.path.join(settings['user_input_folder'], settings['georef_im'])


    # Update water index settings [band 1, band 2, normalised bool]
    # Band list [1 Coastal Blue, 2 Blue, 3 Green I, 4 Green II, 5 Yellow, 6 Red, 7 Red Edge, 8 NIR]
    if settings['water_index'] == 'NDVI':
        # NDVI: (NIR - Red) / (NIR + Red)
        settings['water_index_list'] = [8, 6, 'normalized_difference', '_NDVI.tif']
    elif settings['water_index'] == 'NDWI':
        # NDWI: (Green II - NIR) / (Green II + NIR)
        settings['water_index_list'] = [4, 8, 'normalized_difference', '_NDWI.tif']
    elif settings['water_index'] == 'NDSI':
        # NDSI: (Red - Green II) / (Red + Green II)
        settings['water_index_list'] = [6, 4, 'normalized_difference', '_NDSI.tif']
    elif settings['water_index'] == 'RENDVI':
        # RENDVI: (NIR - Red Edge) / (NIR + Red Edge)
        settings['water_index_list'] = [8, 7, 'normalized_difference', '_RENDVI.tif']
    elif settings['water_index'] == 'Blue_NIR_Ratio':
        # Blue_NIR_Ratio: Blue / NIR
        settings['water_index_list'] = [2, 8, 'ratio', '_Blue_NIR_Ratio.tif']
    elif settings['water_index'] == 'Coastal_NIR_Ratio':
        # Coastal_NIR_Ratio: Coastal Blue / NIR
        settings['water_index_list'] = [1, 8, 'ratio', '_Coastal_NIR_Ratio.tif']
    elif settings['water_index'] == 'Green_RedEdge_Ratio':
        # Green_RedEdge_Ratio: Green II / Red Edge
        settings['water_index_list'] = [4, 7, 'ratio', '_Green_RedEdge_Ratio.tif']
    elif settings['water_index'] == 'Yellow_Red_Ratio':
        # Yellow_Red_Ratio: Yellow / Red
        settings['water_index_list'] = [5, 6, 'ratio', '_Yellow_Red_Ratio.tif']

    # Create AOI polygon from KML file
    settings['aoi_geojson'] = os.path.join(settings['run_input_folder'], settings['site_name'] + '_aoi.geojson')
    gdal.VectorTranslate(settings['aoi_geojson'], gdal.OpenEx(os.path.join(settings['user_input_folder'], settings['aoi_kml'])), format='GeoJSON')
    

    # Scan downloads folder and map relevent contents
    outputs = {}
    outputs['downloads_map'] = map_downloads(settings)


    # output dictionary as log file
    out_file = os.path.join(settings['sl_thresh_ind'], 'input_settings_log_file.csv')
    with open(out_file, 'w') as f:
        for key in settings.keys():
            f.write("%s, %s\n" % (key, settings[key]))

    # print AOI area
    calculate_aoi_area(settings)

    return outputs


def calculate_aoi_area(settings):
    # Load the GeoJSON file
    with open(settings['aoi_geojson'], 'r') as f:
        geojson_data = json.load(f)
    # Extract the first feature's geometry
    polygon = shape(geojson_data['features'][0]['geometry'])
    # Create a transformer from WGS84 (EPSG:4326) to projected EPSG
    transformer = Transformer.from_crs("EPSG:4326", settings['output_epsg'], always_xy=True)
    # Transform the polygon coordinates
    projected_polygon = transform(transformer.transform, polygon)
    # Calculate the area
    area = projected_polygon.area
    print("AOI is", round(area/(1000*1000),1), 'km^2')


def map_downloads(settings):
    ''' Scan download folder and create dictionary of data '''
    
    # Unpack input dictionary
    folder = settings['downloads_folder']
    
    # Remove .aux.xml files that can be generated by QGIS
    for file in os.listdir(folder):
        if file.endswith('.aux.xml'):
            os.remove(os.path.join(folder, file))
    
    # Initialize output
    map_dict = {}
    count = 0
                    
    # Define file endings
    file_endings = [
        '_AnalyticMS_clip.tif',
        '_AnalyticMS_8b_clip.tif',
        '_udm2_clip.tif',
        '_udm_clip.tif',
        '_metadata.json',
        '_metadata_clip.xml'
    ]
                    
    # Initialize dictionary structure using file dates
    for file in os.listdir(folder):
        for end in file_endings:
            if file.endswith(end):
                # Extract date
                filename_parts = file.split('_')
                if len(filename_parts) >= 2:
                    date_str = filename_parts[0]  # 'YYYYMMDD'
                    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    count += 1
                    # Create dictionary for each date
                    if date not in map_dict:
                        map_dict[date] = {}
                        for e in file_endings:
                            map_dict[date][e] = {'filenames': [], 'filepaths': []}
                            if e.startswith('_AnalyticMS'):
                                map_dict[date][e]['timestamp'] = []
                else:
                    print(f"\nUnexpected filename format: {file}")
                break  # Exit the inner loop if a matching ending is found

    # Fill in dictionary with filepath details
    length = len(os.listdir(folder))
    count_file = 0
    for file in os.listdir(folder):
        count_file += 1
        print('\rUpdating Dictionary ' + str(round((count_file / length) * 100, 2)) + '%     ', end='')
        for end in file_endings:
            if file.endswith(end):
                # Extract date
                filename_parts = file.split('_')
                if len(filename_parts) >= 2:
                    date_str = filename_parts[0]  # 'YYYYMMDD'
                    time_str = filename_parts[1]  # 'HHMMSS'
                    date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    # Update dictionary
                    map_dict[date][end]['filenames'].append(file)
                    map_dict[date][end]['filepaths'].append(os.path.join(folder, file))
                    if end.startswith('_AnalyticMS'):
                        # Add timestamp
                        try:
                            hour = int(time_str[0:2])
                            minute = int(time_str[2:4])
                            second = int(time_str[4:6])
                            timestamp = datetime.datetime(
                                year=int(date_str[:4]),
                                month=int(date_str[4:6]),
                                day=int(date_str[6:8]),
                                hour=hour,
                                minute=minute,
                                second=second
                            )
                            map_dict[date][end]['timestamp'].append(timestamp)
                        except ValueError:
                            print(f"\nError parsing time from filename: {file}")
                            map_dict[date][end]['timestamp'].append(None)
                else:
                    print(f"\nUnexpected filename format: {file}")
                break  # Exit the inner loop if a matching ending is found
                                            
    print('\n   ', count, 'images found over', len(map_dict), 'dates')
    
    settings['input_image_count'] = count
    
    return map_dict

