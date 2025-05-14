
import pandas as pd
Dataset = pd.read_csv("C:\\Users\\atang\\OneDrive\\Documents\\Data Science MSc\\Individual Research Project\\Database.csv")
Dataset.head()


#Convert the time column to Datetime for time series analysis
Dataset['Time'] = pd.to_datetime(Dataset['Time'])
Dataset.set_index('Time', inplace=True)
Dataset.head()

# Generate a complete date range
full_range = pd.date_range(start=Dataset.index.min(), end=Dataset.index.max(), freq='5min')

# Identify missing dates
missing_dates = full_range.difference(Dataset.index)

# Print the missing dates
print("Missing dates:")
print(missing_dates)


import matplotlib.pyplot as plt

# Plot the PV_production series to check for stationary and check for time series
plt.figure(figsize=(10, 6))
plt.plot(Dataset['PV_production'])
plt.title('PV Production Over Time')
plt.xlabel('Time')
plt.ylabel('PV Production')
plt.show()


# In[7]:

# Filter data to keep only the year 2020
Dataset_2020 = Dataset[(Dataset.index.year == 2020)]
Dataset_2020.head()
#Dataset_2020.shape


# In[8]:

Dataset_2020 = Dataset_2020.drop(['Wind_production', 'Electric_demand', 'Unnamed: 0'], axis = 1)
print(Dataset_2020.columns)


# In[9]:

Dataset_2020['Wind_speed'] = pd.to_numeric(Dataset_2020['Wind_speed'], downcast='float')
Dataset_2020['Season'] = pd.to_numeric(Dataset_2020['Season'], downcast='float')
Dataset_2020['Day_of_the_week'] = pd.to_numeric(Dataset_2020['Day_of_the_week'], downcast='float')
Dataset_2020['DHI'] = pd.to_numeric(Dataset_2020['DHI'], downcast='float')
Dataset_2020['DNI'] = pd.to_numeric(Dataset_2020['DNI'], downcast='float')
Dataset_2020['GHI'] = pd.to_numeric(Dataset_2020['GHI'], downcast='float')
Dataset_2020['Humidity'] = pd.to_numeric(Dataset_2020['Humidity'], downcast='float')
Dataset_2020['Temperature'] = pd.to_numeric(Dataset_2020['Temperature'], downcast='float')
Dataset_2020['PV_production'] = pd.to_numeric(Dataset_2020['PV_production'], downcast='float')


# In[10]:

#Check if there any negative values within the dataset
# Specify the columns to check
columns_to_check = ['Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI',
       'Wind_speed', 'Humidity', 'Temperature', 'PV_production']

# Check for negative values in each column
for column in columns_to_check:
    negative_values = Dataset_2020[Dataset_2020[column] < 0]
    if not negative_values.empty:
        print(f"Negative values found in column '{column}':")
        print(negative_values)
    else:
        print(f"No negative values found in column '{column}'.")


# In[11]:

# Replace negative PV_production values with zero
Dataset_2020.loc[Dataset_2020['PV_production'] < 0, 'PV_production'] = 0

# In[12]:

# Handle any remaining non-numeric or NA values in PV_production
Dataset_2020 = Dataset_2020.dropna(subset=['PV_production'])
#Dataset_2020.shape


# In[13]:
#Handling missing values 
print(Dataset_2020.isnull().sum())


# In[14]:
#Handling outliers
#calculate and remove outliers
#define a function to calculate and return the outliers
def calculate_outliers(column_series):
    Q1 = column_series.quantile(0.25)
    Q3 = column_series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = column_series[(column_series < lower_bound) | (column_series > upper_bound)]
    return outliers

# Call the function for each column you want to find outliers for and print the results
print(calculate_outliers(Dataset_2020['DHI']))
print(calculate_outliers(Dataset_2020['DNI']))
print(calculate_outliers(Dataset_2020['GHI']))
print(calculate_outliers(Dataset_2020['PV_production']))
print(calculate_outliers(Dataset_2020['Humidity']))
print(calculate_outliers(Dataset_2020['Temperature']))
print(calculate_outliers(Dataset_2020['Wind_speed']))
print(Dataset_2020.shape)


# In[15]:


Dataset.head()


# In[16]:
Dataset_new = Dataset_2020.to_csv('C:\\Users\\atang\\OneDrive\\Documents\\Data Science MSc\\Individual Research Project\\Cleaned Database.csv', index=False)


# In[17]:
import matplotlib.pyplot as plt
# Plot the PV_production series to check for stationary check for time series
plt.figure(figsize=(10, 6))
plt.plot(Dataset_2020['PV_production'])
plt.title('PV Production Over Time')
plt.xlabel('Time')
plt.ylabel('PV Production')
plt.show()


# In[18]:

#Test for stationary within the time series data
#This is to:
from statsmodels.tsa.stattools import adfuller

result = adfuller(Dataset_2020['PV_production'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])


# In[19]:

import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import numpy as np

# Create lag features function (for individual split)
def create_lag_features(data, lags, target_column):
    lagged_data = data.copy()
    for lag in lags:
        lagged_data[f'{target_column}_lag_{lag}'] = lagged_data[target_column].shift(lag)
    return lagged_data.dropna()

# Define lags
lags = [1, 2, 3, 4, 5]

# Initialise TimeSeriesSplit
n_splits = 3
tscv = TimeSeriesSplit(n_splits=n_splits)
r2_scores = []
mse_scores = []

# Cross-validation
for fold, (train_index, test_index) in enumerate(tscv.split(Dataset_2020), 1):
    train_data, test_data = Dataset_2020.iloc[train_index], Dataset_2020.iloc[test_index]
    
    # Create lag features based on training data
    train_data = create_lag_features(train_data, lags, 'PV_production')
    test_data = create_lag_features(test_data, lags, 'PV_production')
    
    # Split into features and target
    X_train = train_data.drop(columns=['PV_production'])
    y_train = train_data['PV_production']
    X_test = test_data.drop(columns=['PV_production'])
    y_test = test_data['PV_production']

    # Model training
    model = XGBRegressor(objective='reg:squarederror')
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    r2_scores.append(r2)
    mse_scores.append(mse)

    # Print individual fold results
    print(f"Fold {fold}: R2 Score = {r2:.4f}, MSE = {mse:.4f}")

# Average metrics
average_r2 = np.mean(r2_scores)
average_mse = np.mean(mse_scores)

# Print average metrics
print(f'\nAverage R2 Score: {average_r2:.4f}')
print(f'Average MSE: {average_mse:.4f}')

# Train final model on the entire dataset
final_data = create_lag_features(Dataset_2020, lags, 'PV_production')
final_features = final_data.drop(columns=['PV_production'])
final_target = final_data['PV_production']

final_model = XGBRegressor(objective='reg:squarederror')
final_model.fit(final_features, final_target)

# Load new dataset for testing
new_dataset = pd.read_csv("C:\\Users\\atang\\OneDrive\\Documents\\Data Science MSc\\Individual Research Project\\Database.csv")

# Convert the time column to datetime for time series analysis
new_dataset['Time'] = pd.to_datetime(new_dataset['Time'])
new_dataset.set_index('Time', inplace=True)

# Filter data for the year 2021
Dataset_2021 = new_dataset[(new_dataset.index.year == 2021)]
Dataset_2021 = Dataset_2021.drop(['Wind_production', 'Electric_demand', 'Unnamed: 0'], axis=1)

columns_to_convert = ['Wind_speed', 'Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 'Humidity', 'Temperature', 'PV_production']
for column in columns_to_convert:
    Dataset_2021[column] = pd.to_numeric(Dataset_2021[column], downcast='float')

# Create lag features for the new dataset
Dataset_2021 = create_lag_features(Dataset_2021, lags, 'PV_production')

# Test only a subset of Dataset_2021 to reduce computer processing time
Dataset_2021 = Dataset_2021.iloc[:10000]

# Split new dataset into features and target
new_features = Dataset_2021.drop(columns=['PV_production'])
new_target = Dataset_2021['PV_production']

# Evaluate final model on the new test set
new_predictions = final_model.predict(new_features)

# Calculate metrics on the new test set
new_r2 = r2_score(new_target, new_predictions)
new_mse = mean_squared_error(new_target, new_predictions)

print(f'\nNew Dataset R2 Score: {new_r2:.4f}')
print(f'New Dataset MSE: {new_mse:.4f}')





#CREATE GUI USING STREAMLIT
import streamlit as st
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt

@st.cache_resource
def load_explainer(_model):
    try:
        # Use TreeExplainer for tree-based models like XGBoost
        return shap.Explainer(_model)
    except UnicodeDecodeError as e:
        st.write(f"SHAP Explainer failed: {e}")
        # Fallback to KernelExplainer for compatibility
        return shap.KernelExplainer(_model.predict, sidebar)

explainer = shap.Explainer(final_model)

# Load explainer
explainer = load_explainer(final_model)

# Create title of page
st.write("""# Solar PV Production Estimator""")

# Create sidebar header
st.sidebar.header('Input PV Parameters')

# User type selection
user_type = st.sidebar.radio("Select Your Role", ("Engineer", "Homeowner"))

# Function to get user inputs
def user_input_parameters(user_type):
    DHI = st.sidebar.number_input("Enter DHI (Diffuse Horizontal Irradiance):", min_value=0.0, max_value=1000.0, value=100.0)
    DNI = st.sidebar.number_input("Enter DNI (Direct Normal Irradiance):", min_value=0.0, max_value=1000.0, value=200.0)
    GHI = st.sidebar.number_input("Enter GHI (Global Horizontal Irradiance):", min_value=0.0, max_value=1000.0, value=300.0)
    Wind_speed = st.sidebar.number_input("Enter Wind Speed (m/s):", min_value=0.0, max_value=100.0, value=5.0)
    Humidity = st.sidebar.number_input("Enter Humidity (%):", min_value=0.0, max_value=100.0, value=50.0)
    Temperature = st.sidebar.number_input("Enter Temperature (°C):", min_value=-10.0, max_value=50.0, value=25.0)
    Season = st.sidebar.selectbox("Select Season:", options=["Winter", "Spring", "Summer", "Autumn"])
    Day_of_the_week = st.sidebar.selectbox("Select Day of the Week:", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    # Default values for lagged features
    default_pv_value = 100.0

    PV_production_lag_1 = st.sidebar.number_input("Enter PV Production Lag 1 (Total PV produced 15mins ago):", min_value=0.0, max_value=10000.0, value=default_pv_value)
    PV_production_lag_2 = st.sidebar.number_input("Enter PV Production Lag 2 (Total PV produced 30mins ago):", min_value=0.0, max_value=10000.0, value=default_pv_value)
    PV_production_lag_3 = st.sidebar.number_input("Enter PV Production Lag 3 (Total PV produced 45mins ago):", min_value=0.0, max_value=10000.0, value=default_pv_value)
    PV_production_lag_4 = st.sidebar.number_input("Enter PV Production Lag 4 (Total PV produced 60mins ago):", min_value=0.0, max_value=10000.0, value=default_pv_value)
    PV_production_lag_5 = st.sidebar.number_input("Enter PV Production Lag 5 (Total PV produced 75mins ago):", min_value=0.0, max_value=10000.0, value=default_pv_value)

    data = {
        'Season': Season,
        'Day_of_the_week': Day_of_the_week,
        'DHI': DHI,
        'DNI': DNI,
        'GHI': GHI,
        'Wind_speed': Wind_speed,
        'Humidity': Humidity,
        'Temperature': Temperature,
        'PV_production_lag_1': PV_production_lag_1,
        'PV_production_lag_2': PV_production_lag_2,
        'PV_production_lag_3': PV_production_lag_3,
        'PV_production_lag_4': PV_production_lag_4,
        'PV_production_lag_5': PV_production_lag_5
    }

    parameters = pd.DataFrame(data, index=[0])

    # Reorder columns to match the order expected by the model
    correct_order = [
        'Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature',
        'PV_production_lag_1', 'PV_production_lag_2', 'PV_production_lag_3', 'PV_production_lag_4', 'PV_production_lag_5'
    ]
    parameters = parameters[correct_order]

    return parameters

# Get sidebar inputs
sidebar = user_input_parameters(user_type)

if sidebar is not None:
    # Convert Season and Day_of_the_week to numerical codes
    sidebar['Season'] = sidebar['Season'].map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3})
    sidebar['Day_of_the_week'] = sidebar['Day_of_the_week'].map({
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6})

    st.subheader('User Input PV Parameters')
    st.write(sidebar)

    # Make prediction
    prediction = final_model.predict(sidebar)
    st.subheader('Predicted PV Production')

    # Create a styled HTML string to display the prediction in a white box, bold, and large
    st.markdown(f"""
        <div style="background-color: white; padding: 20px; border-radius: 10px; text-align: center;">
            <span style="color: black; font-size: 32px; font-weight: bold;">{prediction[0]:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    # Checkbox to toggle SHAP analysis
    show_shap = st.sidebar.checkbox("Show SHAP Analysis")

    if show_shap:
        # SHAP Analysis
        shap_values = explainer.shap_values(sidebar)
        st.subheader('SHAP Analysis')
        st.pyplot(shap.summary_plot(shap_values, sidebar, show=False))

        # Optionally, show SHAP dependence plots
        for feature in sidebar.columns:
            st.pyplot(shap.dependence_plot(feature, shap_values, sidebar, show=False))

# Sidebar for Feedback
st.sidebar.header('Feedback')
feedback = st.sidebar.text_area("Please leave your feedback here.")

# Submit Button
if st.sidebar.button('Submit Feedback'):
    # Handle the feedback submission
    if feedback:
        st.sidebar.success("Thank you for your feedback!")
        # Here you can add code to save or process the feedback, e.g., save to a file or database
        # Example: Save feedback to a text file
        with open('feedback.txt', 'a') as f:
            f.write(feedback + '\n')
    else:
        st.sidebar.error("Please enter some feedback before submitting.")

#make time series predictions
def make_time_series_predictions(model, initial_features, n_steps):
    predictions = []  # To store the predicted values
    current_input = initial_features.copy()  # Start with the initial features

    for step in range(n_steps):  # Loop for the number of steps you want to predict
        prediction = model.predict(np.array(current_input).reshape(1, -1))  # Predict the next step
        predictions.append(prediction[0])  # Store the prediction

        # Update the input for the next prediction (rolling forecast)
        current_input = current_input[1:] + [prediction[0]]  # Shift the inputs to include the new prediction

    return predictions


if user_type == "Engineer":
    # Initial features for the first prediction
    initial_features = [
        sidebar['Season'][0], sidebar['Day_of_the_week'][0], sidebar['DHI'][0], sidebar['DNI'][0], sidebar['GHI'][0],
        sidebar['Wind_speed'][0], sidebar['Humidity'][0], sidebar['Temperature'][0],
        sidebar['PV_production_lag_1'][0], sidebar['PV_production_lag_2'][0], sidebar['PV_production_lag_3'][0],
        sidebar['PV_production_lag_4'][0], sidebar['PV_production_lag_5'][0]
    ]

    # Number of future steps to predict
    n_steps = st.sidebar.slider("Number of steps to predict (e.g., hours):", 1, 24, 12)

    # Perform time-series predictions
    predictions = make_time_series_predictions(final_model, initial_features, n_steps)

    # Display the results
    st.subheader(f'{n_steps}-Step Time-Series PV Production Forecast')

    # Create a time index for the predictions
    time_index = pd.date_range(start=pd.Timestamp.now(), periods=n_steps, freq='H')

    # Create a DataFrame for plotting
    prediction_df = pd.DataFrame({'Time': time_index, 'Predicted PV Production': predictions})
    prediction_df.set_index('Time', inplace=True)

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_index, predictions, label='Predicted PV Production', color='blue', marker='o')
    plt.xlabel('Time')
    plt.ylabel('Predicted PV Production')
    plt.title(f'{n_steps}-Step PV Production Forecast')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()

    # Display the plot in Streamlit
    st.pyplot(plt)

    st.write(prediction_df)  # Display the predictions in tabular form

