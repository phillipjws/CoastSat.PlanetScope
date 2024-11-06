import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import copy
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import os
from sklearn import linear_model

from coastsat_ps.preprocess_tools import create_folder


#%%

def get_closest_datapoint(dates, dates_ts, values_ts):
    """
    Extremely efficient script to get closest data point to a set of dates from a very
    long time-series (e.g., 15-minutes tide data, or hourly wave data)
    
    Make sure that dates and dates_ts are in the same timezone (also aware or naive)
    
    KV WRL 2020

    Arguments:
    -----------
    dates: list of datetimes
        dates at which the closest point from the time-series should be extracted
    dates_ts: list of datetimes
        dates of the long time-series
    values_ts: np.array
        array with the values of the long time-series (tides, waves, etc...)
        
    Returns:    
    -----------
    values: np.array
        values corresponding to the input dates
        
    """
    
    # check if the time-series cover the dates
    if dates[0] < dates_ts[0] or dates[-1] > dates_ts[-1]: 
        raise Exception('Time-series do not cover the range of your input dates')
    
    # get closest point to each date (no interpolation)
    temp = []
    def find(item, lst):
        start = 0
        start = lst.index(item, start)
        return start
    for i,date in enumerate(dates):
        print('\rExtracting closest tide to PS timestamps: %d%%' % int((i+1)*100/len(dates)), end='')
        temp.append(values_ts[find(min(item for item in dates_ts if item > date), dates_ts)])
    values = np.array(temp)
    
    return values


#%% Tidal correction

def tidal_correction(settings, tide_settings, sl_csv):

    # Initialise
    if type(tide_settings['beach_slope']) is list:
        if len(tide_settings['beach_slope']) != len(settings['transects_load'].keys()):
            raise Exception('Beach slope list length does not match number of transects')
    
    # unpack settings
    weight = tide_settings['weighting']
    contour = tide_settings['contour']
    offset = tide_settings['offset']
    mindate = tide_settings['date_min']
    maxdate = tide_settings['date_max']
    
    # import sl data
    sl_csv_tide = copy.deepcopy(sl_csv)
    sl_csv_tide.loc[:,'Date'] = pd.to_datetime(sl_csv_tide.loc[:,'Date'], utc = True)
    
    # Filter by date
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Date'] > pd.to_datetime(mindate, utc = True)]
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Date'] < pd.to_datetime(maxdate, utc = True)]

    # Filter by filter
    sl_csv_tide = sl_csv_tide[sl_csv_tide['Filter'] == 1]

    # Import tide daa
    tide_data = pd.read_csv(os.path.join(settings['user_input_folder'], settings['tide_data']), parse_dates=['dates'])
    dates_ts = [_.to_pydatetime() for _ in tide_data['dates']]
    tides_ts = np.array(tide_data['tide'])
    
    # get tide levels corresponding to the time of image acquisition
    dates_sat = sl_csv_tide['Date'].to_list()
    sl_csv_tide['Tide'] = get_closest_datapoint(dates_sat, dates_ts, tides_ts)
    
    # Perform correction for each transect
    for i, ts in enumerate(settings['transects_load'].keys()):
        # Select beach slope
        if type(tide_settings['beach_slope']) is not list:
            beach_slope = tide_settings['beach_slope']
        else:
            beach_slope = tide_settings['beach_slope'][i]
        
        # Select ts data
        ps_data = copy.deepcopy(sl_csv_tide[['Date',ts, 'Tide']])
        ps_data = ps_data.set_index('Date')
        
        # apply correction
        correction = weight*(ps_data['Tide']-contour)/beach_slope + offset
        sl_csv_tide.loc[:, ts] += correction.values
    
    # Plot tide matching
    fig, ax = plt.subplots(1,1,figsize=(15,4), tight_layout=True)
    ax.grid(which='major', linestyle=':', color='0.5')
    ax.plot(dates_ts, tides_ts, '-', color='0.6', label='all time-series')
    ax.plot(dates_sat, sl_csv_tide['Tide'], '-o', color='k', ms=6, mfc='w',lw=1, label='image acquisition')
    ax.set(ylabel='tide level [m]',xlim=[dates_sat[0],dates_sat[-1]], title='Water levels at the time of image acquisition');
    ax.legend();
    plt.show(block=False)
    plt.savefig(settings['sl_transect_csv'].replace('.csv', '_tide_time_plot.png'), bbox_inches='tight', dpi=300)

    # save csv
    sl_csv_tide = sl_csv_tide.round(2)
    sl_csv_tide.to_csv(settings['sl_transect_csv'].replace('.csv', '_tide_corr.csv'))
    
    return sl_csv_tide


#%% Single transect plot
def ts_plot_single(settings, sl_csv, transect, savgol, x_scale):
    
    # Import PS data and remove NaN values
    ps_data = copy.deepcopy(sl_csv[['Date', transect]])
    ps_data['Date'] = pd.to_datetime(ps_data['Date'], utc=True)
    ps_data.set_index('Date', inplace=True)
    ps_data = ps_data[np.isfinite(ps_data[transect])]
    mean_ps = np.nanmean(ps_data[transect])
    
    # Initialize main figure and axes for timeseries plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
    fig.suptitle(settings['site_name'] + ' Transect ' + transect + ' Timeseries Analysis')
    
    # Timeseries Plot
    ax1.set_title('Original Timeseries')
    ax1.set(ylabel='Chainage [m]', xlabel='Date [UTC]')
    ax1.axhline(y=mean_ps, color='k', linewidth=0.75, label='Mean PS Position', zorder=2)
    ax1.fill_between(ps_data.index, ps_data[transect], y2=mean_ps, alpha=0.25, color='grey', label='PS Data', zorder=3)
    l1, = ax1.plot(ps_data.index, ps_data[transect], linewidth=0.75, alpha=0.6, color='k', label='PS Data', zorder=4)
    
    # Apply Savitzky-Golay filter if requested
    if savgol:
        no_days = (max(ps_data.index) - min(ps_data.index)).days
        if no_days < 16:
            raise Exception('SavGol filter requires >15 days in timeseries')
        
        roll_days = 15
        interp_PL = ps_data.resample('D').mean().interpolate(method='linear')
        interp_PL_sav = signal.savgol_filter(interp_PL[transect], roll_days, 2)
        ax1.plot(interp_PL.index, interp_PL_sav, linewidth=0.75, alpha=0.7, color='r', 
                 label=str(roll_days) + ' Day SavGol Filter', zorder=5)
    
    # Linear trend line (in m/year)
    X = np.array((ps_data.index - ps_data.index[0]).days / 365).reshape(-1, 1)
    y = ps_data[transect].values.reshape(-1, 1)
    model = linear_model.LinearRegression().fit(X, y)
    slope = model.coef_[0][0]
    ax1.plot(ps_data.index, slope * X + model.intercept_, color='blue', linewidth=0.75, linestyle='--', 
             label=f"Trend: {slope:.4f} m/year")
    
    ax1.legend(ncol=3, fontsize='small')
    ax1.grid(visible=True, which='major', linestyle='-')
    
    # Monthly Averages Plot
    monthly_data = ps_data.resample('M').mean()
    # Drop NaNs for linear regression
    monthly_data_clean = monthly_data[transect].dropna()
    monthly_X = np.array((monthly_data_clean.index - monthly_data_clean.index[0]).days / 365).reshape(-1, 1)
    monthly_y = monthly_data_clean.values.reshape(-1, 1)
    
    # Fit linear model only on non-NaN data
    monthly_model = linear_model.LinearRegression().fit(monthly_X, monthly_y)
    monthly_slope = monthly_model.coef_[0][0]
    ax2.set_title('Monthly Averaged Timeseries')
    ax2.plot(monthly_data.index, monthly_data[transect], color='orange', label='Monthly Average', marker='o', linestyle='-')
    ax2.plot(monthly_data_clean.index, monthly_slope * monthly_X + monthly_model.intercept_, color='blue', linestyle='--',
             label=f"Monthly Trend: {monthly_slope:.4f} m/year")
    
    ax2.set(ylabel='Monthly Chainage [m]', xlabel='Date [UTC]')
    ax2.legend(fontsize='small')
    ax2.grid(visible=True, which='major', linestyle='-')
    
    # Seasonal Averages Plot
    seasonal_data = ps_data.resample('Q').mean()  # Quarterly for approximate seasonal representation
    # Drop NaNs for linear regression
    seasonal_data_clean = seasonal_data[transect].dropna()
    seasonal_X = np.array((seasonal_data_clean.index - seasonal_data_clean.index[0]).days / 365).reshape(-1, 1)
    seasonal_y = seasonal_data_clean.values.reshape(-1, 1)
    
    # Fit linear model only on non-NaN data
    seasonal_model = linear_model.LinearRegression().fit(seasonal_X, seasonal_y)
    seasonal_slope = seasonal_model.coef_[0][0]
    ax3.set_title('Seasonal Averaged Timeseries')
    ax3.plot(seasonal_data.index, seasonal_data[transect], color='green', label='Seasonal Average', marker='o', linestyle='-')
    ax3.plot(seasonal_data_clean.index, seasonal_slope * seasonal_X + seasonal_model.intercept_, color='blue', linestyle='--',
             label=f"Seasonal Trend: {seasonal_slope:.4f} m/year")
    
    ax3.set(ylabel='Seasonal Chainage [m]', xlabel='Date [UTC]')
    ax3.legend(fontsize='small')
    ax3.grid(visible=True, which='major', linestyle='-')
    
    # Save plot
    save_folder = os.path.join(settings['sl_thresh_ind'], 'Timeseries Plots')
    os.makedirs(save_folder, exist_ok=True)
    save_file = os.path.join(save_folder, 'transect_' + transect + '_timeseries_analysis.png')
    fig.tight_layout()
    fig.savefig(save_file, dpi=200)
    
    plt.show(block=False)