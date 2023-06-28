import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from math import sqrt
from matplotlib.ticker import MaxNLocator
import pickle

import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'streamlit-lytzen-a8f528cc0316.json'
# from pycaret.regression import setup, compare_models, pull, save_model, predict_model

with st.sidebar:
    st.image("fima_vector.png", width=100)
    st.title("Lytzen Machine Learning")
    choice = st.radio("Menu", ["Data Mesin Lytzen","Quality Control Chart","Machine Learning"])
    # st.info("This application is used for Predict F Value")    

@st.cache_data
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    df = df.sort_values('PV_CYCLE_NO').reset_index(drop=True)
    
    # drop unnecessary columns
    unnecessary_columns = ['id', 'PV_PROCES_TIME_STARTED', 'PV_PROCES_TIME_STOPPED', 'PV_LAST_LEAK_TEST', 'PV_PROCES_TIME_APPR_REJ', 'PV_Leak_Test_Elapsed_HH','PV_Leak_Test_Elapsed_MM',
                      'PV_MAX_STERILIZING_TEMP', 'PV_MIN_STERILIZING_TEMP', 'PV_CIRC_SEC1_ACT_FAULT', 'PV_CIRC_SEC1_ACT_ALARM', 'PV_CIRC_SEC2_ACT_FAULT', 'PV_CIRC_SEC2_ACT_ALARM',
                      'PV_ACTUAL_USER', 'PV_CHAMBER1_TEMP_RAW', 'PV_CHAMBER2_TEMP_RAW', 'PV_INLET_FILTER_dP_RAW', 'PV_CHAMBER_PRESS_RAW', 'PV_STERILE_SIDE_PRESSURE', 'PV_STERILE_SIDE_PRES_RAW',
                      'PV_INT_FILTERS_dP', 'PV_INT_FILTERS_dP_RAW', 'PV_LOAD_TEMP_L1_RAW', 'PV_LOAD_TEMP_L2_RAW', 'PV_LOAD_TEMP_L3_RAW', 'PV_EVENT_TAG_UNLOADING_S', 'PV_AVAILABLE1', 'PV_FILT_TREAT_B_OFF_CYCL',
                      'PV_FILT_TREAT_CL_CYCLES', 'PV_CIRC_SEC1_ACT_SPEED', 'xx', 'PV_LEAK_TEST_ACC_CRIT', 'PV_LEAK_TEST_ACT_W_PRESS', 'PV_LEAK_TEST_END_T_PRESS', 'PV_LEAK_TEST_ST_T_PRESS', 'PV_LEAK_TEST_TIME_ELAP_HPV_LEAK_TEST_TIME_ELAP_H',
                      'PV_LEAK_TEST_TIME_ELAP_M', 'PV_AVAILABLE2', 'PV_STER_TIME_REMAINING', 'PV_CIRC_SEC1_ACT_CURRENT', 'PV_CIRC_SEC1_ACT_TORQUE', 'PV_DELAY_TIME_REMAINING', 'PV_OUTLET_FILTER_dP_RAW',
                      'DATE', 'PV_CYCLE_APPR_OR_REJECT', 'POWER_SUPPLY_FAILURE', 'AL_INFO24_1_NET_COMM_ERR', 'AL_INFO40_0_PILOT_SUPPLY_FAILURE_CHANNEL1', 'AL_INFO40_1_PILOT_SUPPLY_FAILURE_CHANNEL2', 'AL_INFO40_2_PILOT_SUPPLY_FAILURE_CHANNEL3',
                      'AL_INFO40_3_PILOT_SUPPLY_FAILURE_CHANNEL4', 'AL_INFO40_4_EXTERNAL_EXHAUST_FAN_ERROR', 'AL_STOP05_0_EXT_FAN_ERROR', 'CYCLE_TIME_ERROR', 'COMM_UNLOADSIDE_FAILURE', 'ERROR', 'ET', 'FREQ_CONV_COMMS_FAULT', 'FREQ_CONV_FAULT',
                      'I_CIRC_SECT_1_ERROR', 'I_CIRC_SECT_2_ERROR', 'I_EXTERNAL_FAN_ERROR', 'I_HEATER_SECT1_ERROR', 'I_HEATER_SECT2_ERROR', 'I_OVER_PRESS_FAN_ERROR', 'MACHINE_MODE', 'PROCESS_APPROVAL_MODE', 'PROCESS_APPROVAL_STATE', 'PT', 'PV_INLET_FILTER_dP',	'PV_OUTLET_FILTER_dP', 'TIME']
    df = df.drop(columns=unnecessary_columns)
    
    df = df.drop_duplicates()
    
    # convert hours to minutes
    time_columns = ['PV_TOT_COOL_TIME_HH', 'PV_TOT_DELAY_TIME_HH', 'PV_TOT_DRYING_TIME_HH', 
                    'PV_TOT_HEAT_TIME_HH', 'PV_TOT_STERIL_TIME_HH', 'PV_TOTAL_PROCESS_TIME_HH']
    for column in time_columns:
        df[column] = df[column] * 60
    
    # combine hour and minute columns
    combined_time_columns = ['cool_time', 'delay_time', 'drying_time', 'heat_time', 'steril_time', 'total_time']
    for combined, original in zip(combined_time_columns, time_columns):
        df[combined] = df[original] + df[original.replace('_HH', '_MM')]
    
    # rename columns
    renamed_columns = {'PV_CYCLE_NO': 'cycle_number', 'PV_CHAMBER1_TEMP': 'chamber1_temp',
                       'PV_CHAMBER2_TEMP': 'chamber2_temp', 'PV_CHAMBER_PRESS': 'chamber_press', 
                       'PV_LOAD_TEMP_L1': 'load_temp_L1', 'PV_LOAD_TEMP_L2': 'load_temp_L2',
                       'PV_LOAD_TEMP_L3': 'load_temp_L3', 'PV_FH_VALUE': 'fh_value', 'PV_MIN_STERILIZING_TEMP.1' : 'min_sterilizing_temp1'}
    df.rename(columns=renamed_columns, inplace=True)
    
    drop_columns = {'PV_TOT_COOL_TIME_MM', 'PV_TOT_DELAY_TIME_MM', 'PV_TOT_HEAT_TIME_MM', 'PV_TOTAL_PROCESS_TIME_MM', 'PV_TOT_DRYING_TIME_MM', 'PV_TOT_STERIL_TIME_MM'}

    df.drop(columns=drop_columns, inplace=True)

    # drop the original columns
    df = df.drop(columns=time_columns)

    df = df.fillna(method='ffill')
    # Daftar kolom
    columns = df

    # Untuk setiap kolom, hitung IQR dan identifikasi outliers
    for col in columns:
        # Hitung Q1, Q3, dan IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Hitung batas bawah dan atas
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Gantilah outliers dengan median
        df[col] = np.where((df[col] < lower_bound), df[col].mean(), df[col])
        df[col] = np.where((df[col] > upper_bound), df[col].mean(), df[col])
    
    return df



# Use the function
df = preprocess_data('gs://lytzen/vRawDataLytzen23062023.csv')
# Calculate Control Limits
mean_line = df["fh_value"].mean()
upper_line = df["fh_value"].mean() + df["fh_value"].std()
lower_line = df["fh_value"].mean() - df["fh_value"].std()


if choice == "Data Mesin Lytzen":
    st.title("Clean Data after Pre-Processing and EDA")
    st.write(df)

if choice == "Quality Control Chart":
    st.title("Quality Control Chart")

    fh_value_mean = df["fh_value"].mean()
    fh_value_std = df["fh_value"].std()

    upper_line = fh_value_mean + fh_value_std
    lower_line = fh_value_mean - fh_value_std

    cycle_number_max = df["cycle_number"].max()

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=df["cycle_number"], y=df["fh_value"])

    plt.axhline(fh_value_mean, color="red", linestyle="--")
    plt.axhline(upper_line, color="green", linestyle="--")
    plt.axhline(lower_line, color="green", linestyle="--")

    plt.text(cycle_number_max, fh_value_mean, "Mean", va='center', ha="right", bbox=dict(facecolor="w",alpha=0.5),
             fontsize=12, color='black')
    plt.text(cycle_number_max, upper_line, "UCL", va='center', ha="right", bbox=dict(facecolor="w",alpha=0.5),
             fontsize=12, color='black')
    plt.text(cycle_number_max, lower_line, "LCL", va='center', ha="right", bbox=dict(facecolor="w",alpha=0.5),
             fontsize=12, color='black')

    plt.title('X-Bar Control Chart for F-value')
    plt.xlabel('Cycle Number')
    plt.ylabel('F-value')
    st.pyplot(plt.gcf())

    # Write the scores
    st.write("Upper Control Limit (UCL):", upper_line)
    st.write("Lower Control Limit (LCL):", lower_line)

    # Display the top 5 cycle numbers and other column values outside the control limits
    above_ucl = df[df['fh_value'] > upper_line].groupby('cycle_number')['fh_value'].idxmax()
    above_ucl_df = df.loc[above_ucl]

    below_lcl = df[df['fh_value'] < lower_line].groupby('cycle_number')['fh_value'].idxmin()
    below_lcl_df = df.loc[below_lcl]

    if not above_ucl_df.empty:
        st.write("Top value above UCL for each batch number:")
        st.write(above_ucl_df[['cycle_number', 'cool_time', 'delay_time', 'drying_time', 'heat_time', 'steril_time', 'chamber1_temp', 'chamber2_temp', 'chamber_press', 'load_temp_L1', 'load_temp_L2', 'load_temp_L3', 'min_sterilizing_temp1']])

    if not below_lcl_df.empty:
        st.write("Top value below LCL for each batch number:")
        st.write(below_lcl_df[['cycle_number', 'cool_time', 'delay_time', 'drying_time', 'heat_time', 'steril_time', 'chamber1_temp', 'chamber2_temp', 'chamber_press', 'load_temp_L1', 'load_temp_L2', 'load_temp_L3', 'min_sterilizing_temp1']])



# If choice is Machine Learning
if choice == "Machine Learning":
    st.title("Prediction with Machine Learning")

    # Load the trained model
    with open('random_forest_model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)

    # Request the user to input the values for the features
    st.subheader("Please adjust the parameters below:")

    # List of the features that your model has been trained on
    features = df.drop(['fh_value','cycle_number','total_time'], axis=1).columns.tolist()

    # Create a dictionary to store the inputs
    input_dict = {}

    # Get user inputs for each feature
    for i in range(0, len(features), 2):  # step by 2
        col1, col2 = st.columns(2)  # create two columns

        # process for the first column
        feature1 = features[i]
        min_value1 = int(df[feature1].min())
        max_value1 = int(df[feature1].max())

        if min_value1 != max_value1:
            default_value1 = int((min_value1 + max_value1) / 2.0)
            input_value1 = col1.slider(label=f"Adjust {feature1}", min_value=min_value1, max_value=max_value1, value=default_value1)
            input_dict[feature1] = input_value1
        else:
            input_dict[feature1] = min_value1
            col1.text(f"{feature1} unique value: {min_value1}")

        # process for the second column if it exists
        if i+1 < len(features):
            feature2 = features[i+1]
            min_value2 = int(df[feature2].min())
            max_value2 = int(df[feature2].max())

            if min_value2 != max_value2:
                default_value2 = int((min_value2 + max_value2) / 2.0)
                input_value2 = col2.slider(label=f"Adjust {feature2}", min_value=min_value2, max_value=max_value2, value=default_value2)
                input_dict[feature2] = input_value2
            else:
                input_dict[feature2] = min_value2
                col2.text(f"{feature2} unique value: {min_value2}")



    # List of the features that your model has been trained on, in the correct order
    features = ['cool_time', 'delay_time', 'drying_time', 'heat_time', 'steril_time', 'chamber1_temp', 'chamber2_temp', 'chamber_press', 'load_temp_L1', 'load_temp_L2', 'load_temp_L3', 'min_sterilizing_temp1']

    # Once the 'Predict' button has been pressed, make a prediction
    if st.button('Predict'):
        # Convert the dictionary to a dataframe
        input_df = pd.DataFrame([input_dict])

        # Make sure the order of feature is the same as the training data
        input_df = input_df[features]

        # Use the model to make a prediction
        prediction = loaded_model.predict(input_df)

        # Check if the prediction is within the safe range
        if lower_line <= prediction[0] <= upper_line:
            st.success(f"The predicted F-value is {prediction[0]}, which is within the safe range.")
        else:
            st.warning(f"The predicted F-value is {prediction[0]}, which is outside the safe range.")


    # Assume we have X_test (features for the test set) and y_test (true target values for the test set)
    X_test = df[features] 
    y_test = df['fh_value']

    # Predict the target values for the test set
    y_pred = loaded_model.predict(X_test)

    # Calculate metrics
    r2 = r2_score(y_test, y_pred) * 100  # multiply by 100 to convert to percentage
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)  # or mse**(0.5)

    # Round the metrics to remove decimal values
    r2 = round(r2)
    mae = round(mae)
    mse = round(mse)
    rmse = round(rmse)

    # Display the metrics using Streamlit's metric widget
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R^2", f"{r2}%")
    col2.metric("Mean Absolute Error", str(mae))
    col3.metric("Mean Squared Error", str(mse))
    col4.metric("Root Mean Squared Error", str(rmse))

    # Show feature importance
    importance = loaded_model.feature_importances_
    # Create a DataFrame for the feature importances
    feature_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    # Sort by importance
    feature_imp_df = feature_imp_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(10,5))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
    plt.title('Feature Importance')
    plt.tight_layout()
    st.pyplot(plt.gcf())


