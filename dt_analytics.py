import pandas as pd

ri = pd.read_csv("RI-clean.csv.gz", low_memory = False)

ri.head()

ri.shape

ri2 = ri.sample(n = 91741)

ri.columns.values

ri2 = ri2[['state', 'stop_date', 'stop_time', 'driver_gender', 'driver_race_raw', 
'violation_raw', 'violation', 'search_conducted', 'search_type', 'stop_outcome', 'is_arrested', 'stop_duration', 'drugs_related_stop', 'district']]

ri.info()
