import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data (replace this with your actual dataset)
data = {
    'State_Name': ['Maharashtra', 'Maharashtra', 'Punjab', 'Punjab', 'Uttar Pradesh', 'Uttar Pradesh', 'Tamil Nadu', 'Tamil Nadu', 'Bihar', 'Bihar', 'Madhya Pradesh', 'Madhya Pradesh', 'West Bengal', 'West Bengal', 'Gujarat', 'Gujarat', 'Rajasthan', 'Rajasthan', 'Karnataka', 'Karnataka'],
    'District_Name': ['Mumbai', 'Pune', 'Amritsar', 'Ludhiana', 'Agra', 'Kanpur', 'Chennai', 'Coimbatore', 'Patna', 'Gaya', 'Indore', 'Bhopal', 'Kolkata', 'Howrah', 'Ahmedabad', 'Surat', 'Jaipur', 'Jodhpur', 'Bangalore', 'Mysore'],
    'Crop_Year': [2015, 2015, 2016, 2016, 2017, 2017, 2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2024, 2024],
    'Season': ['Kharif', 'Rabi', 'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Zaid', 'Zaid', 'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Kharif', 'Rabi', 'Zaid', 'Zaid', 'Kharif', 'Rabi'],
    'Temperature': [28, 25, 30, 22, 27, 24, 29, 26, 31, 23, 28, 25, 30, 22, 29, 27, 26, 24, 32, 21],
    'Humidity': [70, 60, 65, 75, 68, 72, 74, 62, 80, 77, 69, 63, 71, 76, 68, 73, 65, 70, 64, 72],
    'pH': [6.5, 7.0, 6.8, 6.7, 6.6, 6.9, 7.0, 6.8, 6.7, 7.1, 6.6, 6.5, 6.8, 6.9, 6.7, 7.0, 6.6, 6.7, 6.9, 7.0],
    'Rainfall': [100, 120, 80, 150, 130, 110, 140, 125, 110, 90, 95, 115, 125, 105, 100, 120, 135, 110, 145, 130],
    'Crop': ['Wheat', 'Rice', 'Barley', 'Maize', 'Wheat', 'Rice', 'Cotton', 'Soybean', 'Groundnut', 'Pulses', 'Sugarcane', 'Mustard', 'Rice', 'Potato', 'Tomato', 'Onion', 'Chilies', 'Green Peas', 'Mango', 'Papaya']
}

crop_data = pd.DataFrame(data)

# Streamlit App
st.title('Crop Recommendation System for Farmers')

# Preprocessing Data
# Encode categorical columns
le_state = LabelEncoder()
crop_data['State_Name'] = le_state.fit_transform(crop_data['State_Name'])

le_district = LabelEncoder()
crop_data['District_Name'] = le_district.fit_transform(crop_data['District_Name'])

le_season = LabelEncoder()
crop_data['Season'] = le_season.fit_transform(crop_data['Season'])

le_crop = LabelEncoder()
crop_data['Crop'] = le_crop.fit_transform(crop_data['Crop'])

# Features and Target Variable
X = crop_data[['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y = crop_data['Crop']  # Crop is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model (RandomForestClassifier as an example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy:.2f}")



# Streamlit User Input for Prediction
st.subheader('Enter Crop Data for Recommendation')

# Create input widgets for user input
state_input = st.selectbox('State', le_state.classes_)
district_input = st.selectbox('District', le_district.classes_)
crop_year_input = st.number_input('Crop Year', min_value=2000, max_value=2050, value=2022)
season_input = st.selectbox('Season', le_season.classes_)
temperature_input = st.slider('Temperature (°C)', min_value=10, max_value=40, value=25)
humidity_input = st.slider('Humidity (%)', min_value=30, max_value=100, value=70)
ph_input = st.slider('pH Level', min_value=5.0, max_value=8.5, value=6.5)
rainfall_input = st.slider('Rainfall (mm)', min_value=50, max_value=500, value=100)

# Prepare input data for prediction
input_data = pd.DataFrame([[state_input, district_input, crop_year_input, season_input,
                            temperature_input, humidity_input, ph_input, rainfall_input]],
                          columns=['State_Name', 'District_Name', 'Crop_Year', 'Season', 'Temperature', 'Humidity', 'pH', 'Rainfall'])

# Encode categorical input values for prediction
input_data['State_Name'] = le_state.transform(input_data['State_Name'])
input_data['District_Name'] = le_district.transform(input_data['District_Name'])
input_data['Season'] = le_season.transform(input_data['Season'])

# Make prediction using the trained model
predicted_crop = model.predict(input_data)

# Convert the numerical predicted value back to the original crop name
predicted_crop_name = le_crop.inverse_transform(predicted_crop)

# Display the recommended crop
st.subheader(f'Recommended Crop: {predicted_crop_name[0]}')

# Display the accuracy of prediction based on input conditions
# Calculate model's accuracy for the input crop prediction
input_accuracy = accuracy_score([y_test.iloc[0]], predicted_crop)  # Accuracy based on test data
st.write(f"Accuracy of the Model for Provided Data: {input_accuracy:.2f}")


# Bar chart showing crop recommendation for user input
crop_counts = crop_data['Crop'].value_counts()
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=crop_counts.index, y=crop_counts.values, ax=ax)
ax.set_title('Crop Distribution in Dataset')
ax.set_ylabel('Number of Instances')
st.pyplot(fig)


