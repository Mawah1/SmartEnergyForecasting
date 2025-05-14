Project Overview: This research evaluates the accuracy of machine learning models—specifically XGBoost, LSTM, and Prophet—in predicting SEP using historical production and environmental data. The key objectives are:
•	To compare model performance using metrics such as R-squared scores and MSE across cross-validation folds.
•	To develop a user-friendly GUI that integrates the best-performing model, allowing users to input environmental parameters and obtain real-time SEF.


Data Sources Description: The dataset used in this research focuses on renewable energy production, primarily solar energy. Collected by experts from the CISO (California Independent System Operator ) and NREL (National Renewable Energy Laboratory ), it includes comprehensive environmental and operational parameters crucial for solar energy analysis. PV production data was recorded every 15mins of the day from January 2019 to December 2021. A subset of 105408 records (all data from 2020) was analysed for this research.


Access Instructions for the dataset: available on Mendeley Data - https://data.mendeley.com/datasets/fdfftr3tc2/1

- Download the dataset (CSV file) titled 'Database.csv'
- Save the CSV file on your computer 
- import the file into your virtual environment. If using Python - import pandas as pd
                                                                 - Dataset = pd.read_csv("file path.csv")


Data recreation/preprocessing Steps using pandas in python: 
- Convert time column to Datetime for time series analysis (Dataset['Time'] = pd.to_datetime(Dataset['Time'])
Dataset.set_index('Time', inplace=True))

- Filter out relevant data points (Extract data from 2020 - 'Dataset_2020 = Dataset[(Dataset.index.year == 2020)]')


- Delete irrelevant variables (Dataset_2020 = Dataset_2020.drop(['Wind_production', 'Electric_demand', 'Unnamed: 0'], axis = 1)

- convert all columns to numeric data types (for example: Dataset_2020['DNI'] = pd.to_numeric(Dataset_2020['DNI'], downcast='float'))

- check for and fill in negative values (columns_to_check = ['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI',
       'Wind_speed', 'Humidity', 'Temperature', 'PV_production']

for column in columns_to_check:
    negative_values = Dataset_2020[Dataset_2020[column] < 0]
    if not negative_values.empty:
        print(f"Negative values found in column '{column}':")
        print(negative_values)
    else:
        print(f"No negative values found in column '{column}'.")

- Create lagged features (1 to 5) for non-time-series models.


Sample Data: 

Season	Day_of_the_week	DHI	DNI	GHI	Wind_speed	Humidity	Temperature	PV_production
1	2	0	0	0	1.06	62.808	8.54	0
1	2	0	0	0	1.06	62.814	8.54	0
1	2	0	0	0	1.08	62.928	8.52	0
1	2	0	0	0	1.1	62.998	8.5	0
1	2	0	0	0	1.1	62.89	8.52	0
1	2	0	0	0	1.12	63	8.5	0
1	2	0	0	0	1.12	63.012	8.5	0
1	2	0	0	0	1.12	63.182	8.48	0
1	2	0	0	0	1.14	63.122	8.48	0
1	2	0	0	0	1.14	63.242	8.46	0
1	2	0	0	0	1.14	63.254	8.46	0
1	2	0	0	0	1.2	63.376	8.44	0
1	2	0	0	0	1.2	63.362	8.44	0
1	2	0	0	0	1.2	63.42	8.42	0
1	2	0	0	0	1.2	63.42	8.42	0
1	2	0	0	0	1.2	63.42	8.42	0

