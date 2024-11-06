import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Timedelta
import os

def compare_transect_data(file_1_path, file_2_path, date_tolerance_days=1, rmse_threshold=30):
    # Load the CSV files
    df1 = pd.read_csv(file_1_path)
    df2 = pd.read_csv(file_2_path)

    # Rename date columns for consistency
    df1.rename(columns={'Date': 'dates'}, inplace=True)
    df2.columns = [col.replace("Transect Transect_", "Transect_") for col in df2.columns]

    
    # Convert dates to datetime format and standardize timezones
    df1['dates'] = pd.to_datetime(df1['dates']).dt.tz_localize('UTC')
    df2['dates'] = pd.to_datetime(df2['dates']).dt.tz_convert('UTC')
    
    # Set a tolerance for matching dates
    tolerance = Timedelta(days=date_tolerance_days)

    # Perform a cross join and filter by date difference within the tolerance
    df1_expanded = df1.assign(key=1)
    df2_expanded = df2.assign(key=1)
    crossed_df = pd.merge(df1_expanded, df2_expanded, on='key', suffixes=('_df1', '_df2')).drop(columns='key')
    crossed_df['date_diff'] = (crossed_df['dates_df1'] - crossed_df['dates_df2']).abs()
    tolerant_matches = crossed_df[crossed_df['date_diff'] <= tolerance]

    # Extract transect columns for comparison
    transect_columns = [col for col in df1.columns if col.startswith('Transect_')]
    rmse_results = {}

    # Calculate RMSE for each transect where dates match within tolerance and plot differences
    for transect in transect_columns:
        transect_df1 = f"{transect}_df1"
        transect_df2 = f"{transect}_df2"
        
        # Check if expected columns are in the merged dataframe
        if transect_df1 not in tolerant_matches.columns or transect_df2 not in tolerant_matches.columns:
            print(f"Warning: Expected columns {transect_df1} or {transect_df2} not found in the merged data.")
            continue
        
        # Calculate differences
        tolerant_matches[f"{transect}_diff"] = (
            tolerant_matches[transect_df1] - tolerant_matches[transect_df2]
        )
        
        # Calculate RMSE, ignoring NaN values
        valid_diffs = tolerant_matches[f"{transect}_diff"].dropna()
        if not valid_diffs.empty:
            rmse = np.sqrt(np.mean(valid_diffs ** 2))
            rmse_results[transect] = rmse
            
            # Plot the differences over time
            plt.figure()
            plt.plot(tolerant_matches['dates_df1'], tolerant_matches[f"{transect}_diff"], marker='o', linestyle='-')
            plt.title(f'Difference Plot for {transect}')
            plt.xlabel('Date')
            plt.ylabel('Difference (meters)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

    # Convert RMSE results to a DataFrame
    rmse_df = pd.DataFrame(list(rmse_results.items()), columns=['Transect', 'RMSE'])
    print(rmse_df)

    # Plot RMSE values
    plt.figure(figsize=(10, 6))
    plt.bar(rmse_df['Transect'], rmse_df['RMSE'])
    plt.axhline(y=rmse_threshold, color='r', linestyle='--', label=f'Threshold ({rmse_threshold}m)')
    plt.title('RMSE per Transect Comparison')
    plt.xlabel('Transect')
    plt.ylabel('RMSE')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Filter transects within RMSE threshold
    within_threshold_df = rmse_df[rmse_df['RMSE'] <= rmse_threshold]
    print(f"Transects within {rmse_threshold}m RMSE threshold:\n", within_threshold_df)
    
    return rmse_df, within_threshold_df

# Example usage:
# rmse_df, within_threshold_df = compare_transect_data('file1.csv', 'file2.csv')
rmse_df, within_threshold_df = compare_transect_data(
    r'C:\Users\psteeves\coastal\planetscope_coastsat\outputs\PATRICIA_BAY\shoreline outputs\Global Coreg\NmG\Peak Fraction\PATRICIA_BAY_NmG_Peak Fraction_transect_SL_data.csv', 
    r'C:\Users\psteeves\coastal\CoastSat\data\PATRICIA_BAY\transect_time_series.csv'
)
