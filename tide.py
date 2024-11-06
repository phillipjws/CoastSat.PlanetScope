#%%
import pyfes
import os
import pytz
import datetime
import csv
import numpy as np
import matplotlib.pyplot as plt
from coastsat_ps import SDS_slope
from sklearn.metrics import root_mean_squared_error

#%% Load FES2022 configuration
print('Loading fes2022')
config_filepath = os.pardir
config = os.path.join(config_filepath, 'fes2022.yaml')
handlers = pyfes.load_config(config)
ocean_tide = handlers['tide']
load_tide = handlers['radial']

#%%
centroid = [-132.97826548310042,
          69.44147810360226]
centroid[0] = centroid[0] + 360
date_range = [pytz.utc.localize(datetime.datetime(2014, 1, 1)), pytz.utc.localize(datetime.datetime(2014, 2, 1))]
timestep = 60 * 60  # 1 hour in seconds

dates_ts, tides_ts = SDS_slope.compute_tide(centroid, date_range, timestep, ocean_tide, load_tide)

# Adjust the predicted tide with the given offset
offset = 0.4
predicted_tides = np.array(tides_ts) + offset

dates_csv = []
observed_tides = []
path = os.path.join(os.getcwd(), 'tide_csvs', '6485-01-JAN-2014_slev.csv')
with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)  # Skip header
    for row in reader:
        if len(row) >= 2:
            timestamp = datetime.datetime.strptime(row[0], '%Y-%m-%d %H:%M')
            dates_csv.append(pytz.utc.localize(timestamp))
            observed_tides.append(float(row[1]))

aligned_observed = []
aligned_predicted = []
aligned_dates = []

for date, pred_tide in zip(dates_ts, predicted_tides):
    if date in dates_csv:
        index = dates_csv.index(date)
        aligned_observed.append(observed_tides[index])
        aligned_predicted.append(pred_tide)
        aligned_dates.append(date)

# Calculate RMSE
rmse = root_mean_squared_error(aligned_observed, aligned_predicted)
print(f"RMSE between observed and predicted water levels: {rmse:.3f} metres")

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(aligned_dates, aligned_observed, label="Observed Water Level", color="blue", marker="o", linestyle="--")
plt.plot(aligned_dates, aligned_predicted, label="Predicted Water Level", color="orange", marker="x", linestyle="-")
plt.xlabel("Date and Time (UTC)")
plt.ylabel("Water Level (metres)")
plt.title("Observed vs Predicted Water Levels in Tuktoyaktuk With Offset 0.4m")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()
# %%
