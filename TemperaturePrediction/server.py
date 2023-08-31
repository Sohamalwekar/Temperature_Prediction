from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Convert sunrise and sunset time strings to numerical values
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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        hour = int(request.form['hour'])
        day = int(request.form['day'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        
        new_data = pd.DataFrame({
            'mintempC': [26],
            'maxtempC': [36],
            'precipMM': [0],
            'hour': [hour],
            'day': [day],
            'month': [month],
            'year': [year]
        })
        
        # Load the trained models
        sunrise_model = pickle.load(open("sunrise_model.pkl", "rb"))
        sunset_model = pickle.load(open("sunset_model.pkl", "rb"))
        humidity_model = pickle.load(open("humidity_model.pkl", "rb"))
        temperature_model = pickle.load(open("temperature_model.pkl", "rb"))
        
        # Make predictions
        sunrise_prediction = sunrise_model.predict(new_data)
        sunset_prediction = sunset_model.predict(new_data)
        humidity_prediction = humidity_model.predict(new_data)
        temperature_prediction = temperature_model.predict(new_data)
        
        # Convert predictions to readable format
        sunrise_prediction = [time + " AM" for time in minutes_to_time(sunrise_prediction)]  # Add "AM" after each sunrise time
        sunset_prediction = [time + " PM" for time in minutes_to_time(sunset_prediction)]  # Add "PM" after each sunset time
        humidity_prediction = format(humidity_prediction[0], '.2f') +'%'  # Format humidity to 2 decimal places
        temperature_prediction = format(temperature_prediction[0], '.2f') + "°C"  # Format temperature to 2 decimal places and add °C symbol
        
        prediction = {
            'sunrise': sunrise_prediction,
            'sunset': sunset_prediction,
            'humidity': humidity_prediction,
            'temperature': temperature_prediction
        }
        
        return render_template('index.html', prediction=prediction)
    
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
