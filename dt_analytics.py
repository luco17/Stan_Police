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

# Create a DataFrame of stops in which a search was conducted
searched = ri[(ri.search_conducted == True)]

# Calculate the overall frisk rate by taking the mean of 'frisk'
print(searched.frisk.mean())

# Calculate the frisk rate for each gender
print(searched.groupby(['driver_gender']).frisk.mean())

# Calculate the overall arrest rate
print(ri.is_arrested.mean())

# Calculate the hourly arrest rate
print(ri.groupby([ri.index.hour]).is_arrested.mean())

# Save the hourly arrest rate
hourly_arrest_rate = ri.groupby([ri.index.hour]).is_arrested.mean()

# Import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Create a line plot of 'hourly_arrest_rate'
hourly_arrest_rate.plot()

# Add the xlabel, ylabel, and title
plt.xlabel('Hour')
plt.ylabel('Arrest Rate')
plt.title('Arrest Rate by Time of Day')

# Display the plot
plt.show()

# Calculate the annual rate of drug-related stop
print(ri.drugs_related_stop.resample('A').mean())

# Save the annual rate of drug-related stops
annual_drug_rate = ri.drugs_related_stop.resample('A').mean()

# Create a line plot of 'annual_drug_rate'
annual_drug_rate.plot()

# Display the plot
plt.show()

#Converting the data type of 'search_conducted' to allow resample operations
ri['search_conducted'] = ri.search_conducted.astype('bool')

# Calculate and save the annual search rate
annual_search_rate = ri.search_conducted.resample('A').mean()

# Concatenate 'annual_drug_rate' and 'annual_search_rate'
annual = pd.concat([annual_drug_rate, annual_search_rate], axis = 'columns')

# Create subplots from 'annual'
annual.plot(subplots = True)

# Display the subplots
plt.show()
# Create a frequency table of districts and violations (using pd.crosstab)
print(pd.crosstab(ri.district, ri.violation))

# Save the frequency table as 'all_zones'
all_zones = pd.crosstab(ri.district, ri.violation)

# Select rows 'Zone K1' through 'Zone K3'
print(all_zones.loc['Zone K1': 'Zone K3'])

# Save the smaller table as 'k_zones'
k_zones = all_zones.loc['Zone K1': 'Zone K3']

# Create a stacked bar plot of 'k_zones'
k_zones.plot(kind = 'bar', stacked = True)

# Display the plot
plt.show()

#changetype doesn't work with certain types of data, accordingly a custom change is executed using .map
# Print the unique values in 'stop_duration'
print(ri.stop_duration.unique())

# Create a dictionary that maps strings to integers
mapping = {"0-15 Min":8, "16-30 Min":23, "30+ Min":45}

# Convert the 'stop_duration' strings to integers using the 'mapping'
ri['stop_minutes'] = ri.stop_duration.map(mapping)

# Print the unique values in 'stop_minutes'
print(ri.stop_minutes.unique())

# Calculate the mean 'stop_minutes' for each value in 'violation_raw'
print(ri.groupby(['violation_raw']).stop_minutes.mean())

# Save the resulting Series as 'stop_length'
stop_length = ri.groupby(['violation_raw']).stop_minutes.mean()

# Sort 'stop_length' by its values and create a horizontal bar plot
stop_length.sort_values().plot(kind = 'barh')

# Display the plot
plt.show()
