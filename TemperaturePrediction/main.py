import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
data = pd.read_csv("MUMBAI.csv", index_col='date_time')
# Prepare the data

X = data[['mintempC', 'maxtempC', 'precipMM']]  # Features INDEPENDENT
y_sunrise = data['sunrise']  # Sunrise target DEPENDENT VARIABLE
y_sunset = data['sunset']  # Sunset target
y_humidity = data['humidity']  # Humidity target
y_temperature = data['tempC']  # Temperature target

X.index = pd.to_datetime(X.index)

def create_feature(X):
    X= X.copy()
    X['hour'] = X.index.hour
    X['day'] = X.index.day
    X['month'] = X.index.month
    X['year'] = X.index.year
    return X


X = create_feature(X)
# Convert sunrise and sunset time strings to numerical values
def time_to_minutes(time_str):
    time_parts = time_str.split(':')
    hours = int(time_parts[0])
    minutes = int(time_parts[1][:2])  # Extracting minutes and ignoring AM/PM
    total_minutes = hours * 60 + minutes
    return total_minutes

def minutes_to_time(minutes):
    if isinstance(minutes, np.ndarray):
        hours = (minutes // 60).astype(int)
        minutes = (minutes % 60).astype(int)
        time_str = [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]
        return time_str
    else:
        hours = int(minutes) // 60
        minutes = int(minutes) % 60
        time_str = f"{hours:02d}:{minutes:02d}"
        return time_str



y_sunrise = y_sunrise.apply(time_to_minutes)
y_sunset = y_sunset.apply(time_to_minutes)

# Split the data into training and testing sets
X_train, X_test, y_sunrise_train, y_sunrise_test, y_sunset_train, y_sunset_test, y_humidity_train, y_humidity_test, y_temperature_train, y_temperature_test = train_test_split(X, y_sunrise, y_sunset, y_humidity, y_temperature, test_size=0.2, random_state=42)

# Sunrise prediction
sunrise_model = LinearRegression()
sunrise_model.fit(X_train, y_sunrise_train)# traing the model by train
# Sunset prediction
sunset_model = LinearRegression()
sunset_model.fit(X_train, y_sunset_train)

# Humidity prediction
humidity_model = LinearRegression()
humidity_model.fit(X_train, y_humidity_train)

# Temperature prediction
temperature_model = LinearRegression()
temperature_model.fit(X_train, y_temperature_train)


# Predict on the test data
y_sunrise_pred = sunrise_model.predict(X_test)
y_sunset_pred = sunset_model.predict(X_test)
y_humidity_pred = humidity_model.predict(X_test)
y_temperature_pred = temperature_model.predict(X_test)



# Create a new data frame with the input values
new_data = pd.DataFrame({
    'mintempC': [20],
    'maxtempC': [30],
    'precipMM': [0.5],
    'hour': [15],
    'day': [10],
    'month': [6],
    'year': [2023]
})

# Make predictions using the trained models
new_sunrise_prediction = sunrise_model.predict(new_data)
new_sunset_prediction = sunset_model.predict(new_data)
new_humidity_prediction = humidity_model.predict(new_data)
new_temperature_prediction = temperature_model.predict(new_data)


#print(new_sunrise_prediction)
#print(new_sunset_prediction)
#print(new_humidity_prediction)
#print(new_temperature_prediction)

import pickle
with open("temperature_model.pkl","wb") as f:
    pickle.dump(temperature_model ,f)

with open("sunrise_model.pkl","wb") as f:
    pickle.dump(sunrise_model  ,f)

with open("sunset_model.pkl","wb") as f:
    pickle.dump(sunset_model ,f)

with open("humidity_model.pkl","wb") as f:
    pickle.dump(humidity_model,f)