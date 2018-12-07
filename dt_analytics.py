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

###Initial EDA###
# Count the unique values in 'violation'
print(ri.violation.value_counts())

# Express the counts as proportions
print(ri.violation.value_counts(normalize = True))

# Create a DataFrame of female drivers
female = ri[ri['driver_gender'] == "F"]

# Create a DataFrame of male drivers
male = ri[ri['driver_gender'] == "M"]

# Compute the violations by female drivers (as proportions)
print(female.violation.value_counts(normalize = True))

# Compute the violations by male drivers (as proportions)
print(male.violation.value_counts(normalize = True))

# Create a DataFrame of female drivers stopped for speeding
female_and_speeding = ri[(ri.driver_gender == "F") & (ri.violation == "Speeding")]

# Create a DataFrame of male drivers stopped for speeding
male_and_speeding = ri[(ri.driver_gender == "M") & (ri.violation == "Speeding")]

# Compute the stop outcomes for female drivers (as proportions)
print(female_and_speeding.stop_outcome.value_counts(normalize = True))

# Compute the stop outcomes for male drivers (as proportions)
print(male_and_speeding.stop_outcome.value_counts(normalize = True))

# Check the data type of 'search_conducted'
print(ri.search_conducted.dtype)

# Calculate the search rate by counting the values
print(ri.search_conducted.value_counts(normalize = True))

# Calculate the search rate by taking the mean
print(ri.search_conducted.mean())

# Calculate the search rate for both groups simultaneously
print(ri.groupby(['driver_gender']).search_conducted.mean())

# Reverse the ordering to group by violation before gender
print(ri.groupby(['violation', 'driver_gender']).search_conducted.mean())

# Count the 'search_type' values
print(ri.search_type.value_counts())

# Check if 'search_type' contains the string 'Protective Frisk'
ri['frisk'] = ri.search_type.str.contains('Protective Frisk', na = False)

# Check the data type of 'frisk'
print(ri.frisk.dtype)

# Take the sum of 'frisk'
print(ri.frisk.sum())
