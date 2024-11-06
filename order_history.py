total_quota = 42857.0

quota_used = 0

"""
This is just the number under the 'estimated area' for each one of my orders.
Not too sure if this is what you were looking for, I can get the area of each AOI I used aswell.
*Note the 42857 km^2 number is just what Roger has told me is dedicated for our specific project seperate from the general NRCan quota.
"""

orders = {
    "Noble_Creek-July-30-2020-May-21-2021": 80.575,
    "Noble_Creek-May-21-2021-March-5-2022": 167.664,
    "Noble_Creek-March-5-2022-November-18-2022": 167.529,
    "Noble_Creek-November-18-2022-September-15-2023": 167.773,
    "Noble_Creek-September-15-2023-April-21-2024": 129.238,
    "Noble_Creek-April-22-2024-November-1-2024": 167.862,
    "Patricia_Bay-August-28-2020-March-4-2022": 301.688,
    "Patricia_Bay-March-5-2022-July-20-2023": 397.081,
    "Patricia_Bay-July-21-2023-October-24-2024": 394.157,
    "Island_View_Beach-July-30-2020-August-17-2022": 177.332,
    "Island_View_Beach-August-18-2022-July-19-2023": 311.212,
    "Island_View_Beach-July-20-2023-October-30-2024": 311.454,
    "Lower_Sidney_Island-July-30-2020-September-22-2020": 89.046,
    "Lower_Sidney_Island-October-24-2020-March-22-2023": 1780.267,
    "Lower_Sidney_Island-March-29-2023-July-11-2024": 1778.411,
    "Lower_Sidney_Island-July-13-2024-October-24-2024": 587.314,
    "Upper_Sidney_Island-July-3-2020-September-2-2021": 797.137,
    "Saanich_Peninsula": 913.418,
    "Upper_Sidney_Island-September-9-2021-August-2-2023": 1563.063,
    "Upper_Sidney_Island-August-4-2023-present": 1469.015,
    "Lower_James_Island-July-30-2020-August-25-2021": 255.398,
    "Lower_James_Island-August-29-2021-July-21-2023": 456.013,
    "Lower_James_Island-July-27-2023-Present": 456.048,
    "Basemap": 886.213,
    "James_Island-August-28-2020-August-4-2021": 203.344,
    "James_Island_August-10-2021-July-19-2023": 398.609,
    "James_Island_July-20-2023-present": 398.709,
    "South_Saanich": 88.836,
    "James_Island": 520.104,
    "Cordova_Bay": 117.658,
    "Patricia_Bay_24-25": 134.804,
    "Pat_Bay_4band": 74.735,
    "Pat Bay Test": 44.001
}

for key in orders:
    quota_used += orders[key]

print(f"Quota used: {quota_used:,.3f} km^2, Remaining quota: {total_quota - quota_used:,.3f} km^2, Total quota: {total_quota} km^2")
print(f"You have used {(quota_used / total_quota) * 100:.1f}% of the total quota")

"""
Output:
Quota used: 15,785.708 km^2, Remaining quota: 27,071.292 km^2, Total quota: 42857.0 km^2
You have used 36.8% of the total quota
"""