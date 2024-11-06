#%% CoastSat for PlanetScope Dove Imagery

# load coastsat modules
from coastsat_ps.data_import import initialise_settings
from coastsat_ps.extract_shoreline import extract_shorelines, compute_intersection
from coastsat_ps.interactive import filter_shorelines                    
from coastsat_ps.preprocess import (data_extract, pre_process, select_ref_image, 
                                    add_ref_features)
from coastsat_ps.postprocess import tidal_correction, ts_plot_single


#%% 0) User Input Settings 

settings = {
    
    ### General Settings ###
    # Site name (for output folder and files) 
    'site_name': 'PATRICIA_BAY',
    # Maximum image cloud cover percentage threshold
    'cloud_threshold': 2, # Default 10
    # Minimum image AOI cover percentage threshold
    'extent_thresh': 95, # Default 80
    # Desired output shoreline epsg
    'output_epsg': '3157',

    ### Reference files (in "...CoastSat.PlanetScope/user_inputs/") ###
    # Area of interest file (save as .kml file from geojson.io website)
    'aoi_kml': 'PATRICIA_BAY_polygon.kml',
    # Transects in geojson file (ensure same epsg as output_epsg)
    'transects': False, # False
        # If False boolean given, popup window will allow for manual drawing of transects
    # Tide csv file in MSL and UTC 
    'tide_data': 'patricia_bay_tides.csv',
    # Local folder planet imagery downloads location (provide full folder path)
    'downloads_folder': r'C:\Users\psteeves\coastal\planetscope_coastsat\PS_Imagery\Patricia_Bay_24-25_psscene_analytic_8b_udm2\PSScene',

    ### Processing settings ###
    # Machine learning classifier filename (in "...CoastSat.PlanetScope/classifier/models")
        # A new classifier may be re-trained after step 1.3. Refer "...CoastSat.PlanetScope/classifier/train_new_classifier.py" for instructions. 
    'classifier': 'NN_8b.pkl',
    # Image co-registration choice ['Coreg Off', 'Local Coreg', 'Global Coreg']
    'im_coreg': 'Local Coreg', # refer https://pypi.org/project/arosics/ for details on Local vs Global coreg. Local recommended but slower. 
    # Coregistration land mask - when set to False, a new land mask is calculated for each image (slower but more accurate for large geolocation errors or where the land area changes significantly)
    'generic_land_mask': False,

    ### Advanced settings ###
    # Buffer size around masked cloud pixels [in metres]
    'cloud_buffer': 9, # Default 9 (3 pixels)  
    # Max distance from reference shoreline for valid shoreline [in metres]
    'max_dist_ref': 150, # Default 75
    # Minimum area (m^2) for an object to be labelled as a beach
    'min_beach_area': 50*50, # Default 22500
    # Minimum length for identified contour line to be saved as a shoreline [in metres]
    'min_length_sl': 250, # Default 500 
    # GDAL location setting (Update path to match GDAL path. Update 'coastsat_ps' to chosen environment name. Example provided is for mac)
    'GDAL_location': r'C:\Users\psteeves\AppData\Local\miniforge3\envs\coastsat_ps\Library\bin',
        # for Windows - Update 'anaconda2' to 'anaconda3' depending on installation version.
        # 'GDAL_location': r'C:\ProgramData\Anaconda3\envs\coastsat_ps\Library\bin',

    #### Additional advanced Settings can be found in "...CoastSat.PlanetScope/coastsat_ps/data_import.py"

    }


# Import data and updade settings based on user input
outputs = initialise_settings(settings)


#%% 1.1) Pre-processing - TOA conversion and mask extraction

data_extract(settings, outputs)


#%% 1.2) Pre-processing - Select reference image for co-registration

select_ref_image(settings,
                 # set as true to replace previously selected ref_im
                 replace_ref_im = True)

# If the land mask region is poor, try selecting another reference image by setting replace_ref_im = True

# If the land mask covers thin land regions (ie barrier islands), try adjusting the following settings:
#    - reduce min_beach_area (in cell 0)
#    - reduce land_mask_smoothing_1 and 2 (in data_import.py)

# If the land mask it still poor, try retraining the classifier for your site to improve pixel classification


#%% 1.3) Pre-Processing - image coregistration and scene merging

# raise Exception('Run cell 1.3 manually')
    
# Due to spyder issue, select the below code and press F9 to run rather than running individual cell
pre_process(settings, outputs, 
        # del_files_int = True will delete intermediate coregistration files to save space
        del_files_int = False,
        # set as true to replace previously run preprocessing
        rerun_preprocess = True)

# If coregistration is performing poorly, the following may help:
#    - try a new reference image
#    - reduce tie-point grid size and tie-point window size (in data_import.py)
#    - adjust the 'generic_land_mask' switch
#    - adjust the coregistration method (global or local)


#%% 2.1) Select georectified/merged image for classification, reference shoreline and transects

add_ref_features(settings, plot = False, redo_features = False)


#%% 2.2) Extract shoreline data
    
# Note that output shoreline .geojson file for use in GIS software is not todally corrected
    
shoreline_data = extract_shorelines(outputs, settings, 
                                    
        # del_index = True will delete water index .tif files once used to save space
        del_index = True, 

        # set as true to replace previously extracted shorelines
        rerun_shorelines = True,

        # reclassify = True will reclassify images if they have been classified previously
            # useful when running again with a new classifier
            # use False to save time on re-runs with the same classifier to save processing time
        reclassify = True)
 
 # Plot parameters and layout can be adjusted in ...coastsat_ps/plotting.py


#%% 3) Manual error detection

    # Option 1 - All images pass, creates a csv in the outputs folder: ...CoastSat.PlanetScope/outputs/SITE/shoreline outputs/COREG/NmB/Peak Fraction/shoreline_filter.csv"
        # manual_filter & load_csv = False

    # Option 2 - DEFAULT - popup window to keep or discard images (saves choices as a csv to file from option 1):
        # manual_filter = True & load_csv = False    

    # Option 3 - loads and applies the csv saved from option 1 or 2. This file can be manually updated with a text editor prior to running. 
        # manual_filter = False & load_csv = True
         
shoreline_data = filter_shorelines(settings,
        manual_filter = True, load_csv = False)


#%% 4) Shoreline transect intersction and csv export

sl_csv = compute_intersection(shoreline_data, settings)


#%% 5) Tidal Correction & filtering

tide_settings = {
    # select beach slope as a generic value, or list of values corresponding to each transect
        # Transect specific beach slope values can be extracted with the CoastSat beach slope tool https://github.com/kvos/CoastSat.slope
    'beach_slope': [0.085, 0.075, 0.08, 0.08, 0.1], #0.1 - Can be found using CoastSat.Slope toolbox
    
    # Reference elevation contour
    'contour': 0.7,
    # Tidal correction weighting
    'weighting': 1,
    # Offset correction (+ve value corrects sl seaward, ie. increases chainage)
    'offset': 0,
    
    # Date filter (minimum)
    'date_min':'2014-01-01',
    # Date filter (maximum)
    'date_max':'2023-12-31' 
    }

sl_csv_tide = tidal_correction(settings, tide_settings, sl_csv)


#%% 6) Plot transects
    
for transect in settings['transects_load'].keys():
    ts_plot_single(settings, sl_csv_tide, transect, 
                   
        # set savgol = True to plot 15 day moving average shoreline position
        # Requires > 15 day shorleine timeseries range
        savgol = True,
        
        # set x_scale for x-axis labels ['days', 'months', 'years']
        x_scale = 'years')
    
  
#%% Approximate times (for ~1000 downloaded images)
    # 1.1) 20min
    # 1.3) 2.5h coregistration, 35min merge
    # 2.2) 1h classification, 50min shoreline extraction

