"""
===============================================================================
Regression Project: Anticipate Electricity Consumption Needs of
Buildings - Exploratory Analysis
===============================================================================

This file is organised as follows:
1. Load and explore raw datasets
2. Selection of the final dataset
3. Cleanse and save the dataset
"""
# Standard libraries
import platform

# Other libraries
import pandas as pd
import sweetviz as sv
import ydata_profiling


from sweetviz import analyze
from ydata_profiling import ProfileReport
from functions import *


# Display versions of platforms and packages
print('\n\nPython: {}'.format(platform.python_version()))
print('Pandas: {}'.format(pd.__version__))
print('Sweetviz: {}'.format(sv.__version__))
print('YData-profiling: {}'.format(ydata_profiling.__version__))



# Constants
MAX_ROWS_DISPLAY = 100
MAX_COLUMNS_DISPLAY = 150

# Set the maximum number of rows and columns to display by Pandas
pd.set_option('display.max_rows', MAX_ROWS_DISPLAY)
pd.set_option('display.max_columns', MAX_COLUMNS_DISPLAY)



"""
===============================================================================
1. Load and explore raw datasets
===============================================================================
"""
print(f'\n\n\n1. Load and explore raw datasets')


# Paths of datasets
INPUT_CSV_1 = 'datasets/2015-building-energy-benchmarking.csv'
INPUT_CSV_2 = 'datasets/2016-building-energy-benchmarking.csv'


# Load 2015 building energy benchmarking dataset
print(f'\n\nLoad 2015 building energy benchmarking dataset:')
raw_dataset_2015 = load_dataset(file_path=INPUT_CSV_1, encoding='utf-8')

# Display 2015 dataset information and description
dataset_info_description(dataset=raw_dataset_2015, max_rows=15)

# Generate 2015 dataset report
raw_dataset_2015_report = analyze(source=raw_dataset_2015)
raw_dataset_2015_report.show_html('datasets/raw_dataset_2015_report.html')


# Load 2016 building energy benchmarking dataset
print(f'\n\nLoad 2016 building energy benchmarking dataset:')
raw_dataset_2016 = load_dataset(file_path=INPUT_CSV_2, encoding='utf-8')

# Display 2016 dataset information and description
dataset_info_description(dataset=raw_dataset_2016, max_rows=15)

# Generate 2016 dataset report
raw_dataset_2016_report = analyze(source=raw_dataset_2016)
raw_dataset_2016_report.show_html('datasets/raw_dataset_2016_report.html')



"""
===============================================================================
2. Selection of final datasets
===============================================================================
"""
print(f'\n\n\n2. Selection of final datasets')


# Create new relevant features
dataset_2015 = raw_dataset_2015.copy()
dataset_2016 = raw_dataset_2016.copy()

# Create the feature 'BuildingAge' for heating degree days
dataset_2015['BuildingAge'] = (
    dataset_2015['DataYear'] - dataset_2015['YearBuilt']
)
dataset_2016['BuildingAge'] = (
    dataset_2016['DataYear'] - dataset_2016['YearBuilt']
)

# Select common features to both datasets
dataset_2015 = dataset_2015.rename(
    columns={'GHGEmissions(MetricTonsCO2e)': 'TotalGHGEmissions'})
features = list(set(dataset_2015.columns).intersection(
    set(dataset_2016.columns)))
dataset = pd.concat([dataset_2015[features], dataset_2016[features]])

# Display dataset information and description
dataset_info_description(dataset=dataset, max_rows=15)



"""
===============================================================================
3. Cleanse and save the dataset
===============================================================================
"""
print(f'\n\n\n3. Cleanse and save datasets')


unrelevant_features = [
    'OSEBuildingID', 'TaxParcelIdentificationNumber', 'CouncilDistrictCode',
    'DefaultData', 'ComplianceStatus', 'YearBuilt', 'DataYear', 'Outlier',
    'PropertyName'
]
leakage_features = [
    'Electricity(kBtu)', 'Electricity(kWh)', 'NaturalGas(kBtu)',
    'NaturalGas(therms)', 'SiteEnergyUseWN(kBtu)', 'SiteEUI(kBtu/sf)',
    'SiteEUIWN(kBtu/sf)', 'SourceEUI(kBtu/sf)', 'SourceEUIWN(kBtu/sf)',
    'SteamUse(kBtu)'
]
dataset = dataset.drop(unrelevant_features + leakage_features, axis=1)


# Management of the completion rate
dataset = dataset.dropna(axis=1, thresh=int(0.7 * len(dataset)))

# Display dataset information and description
dataset_info_description(dataset=dataset, max_rows=15)


# Management of duplicates
print('\n\nManagement of duplicates:')
dataset_duplicate = dataset[dataset.duplicated()]
print('\nDimensions of dataset duplicate: {}'.format(dataset_duplicate))

# Display dataset information and description
dataset_info_description(dataset=dataset, max_rows=15)

# Generate dataset report
dataset_report_sv = analyze(source=dataset)
dataset_report_sv.show_html('datasets/dataset_report_sv.html')
dataset_report_ydp = ProfileReport(df=dataset, title='Dataset Report')
dataset_report_ydp.to_file('datasets/dataset_report_ydp.html')

# Save dataset in CSV format
dataset.to_csv('datasets/dataset.csv', index=False)
