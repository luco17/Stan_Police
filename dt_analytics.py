import pandas as pd, os, numpy as np, matplotlib.pyplot as plt

ri_raw = pd.read_csv("RI-clean.csv.gz", low_memory = False)

ri_raw.shape

ri = ri_raw.sample(n = 91741)

ri = ri[['stop_date', 'stop_time', 'driver_gender', 'driver_race',
'violation_raw', 'violation', 'search_conducted', 'search_type', 'stop_outcome', 'is_arrested', 'stop_duration', 'drugs_related_stop', 'district']]

ri.info()

print(ri.isnull().sum())

ri.dropna(subset = ['driver_gender'], inplace = True)

ri['is_arrested'] = ri.is_arrested.astype('bool')

combined = ri.stop_date.str.cat(ri.stop_time, sep = ' ')

ri['stop_datetime'] = pd.to_datetime(combined)

# Set 'stop_datetime' as the index
ri.set_index('stop_datetime', inplace = True)
