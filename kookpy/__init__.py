import requests
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import joblib
from datetime import datetime, timedelta
import os
import numpy as np
import streamlit as st
import sqlite3
import bcrypt

# base urls for the open-meteo apis
GEOCODING_API_URL = "https://geocoding-api.open-meteo.com/v1/search"
MARINE_API_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_WEATHER_API_URL = "https://archive-api.open-meteo.com/v1/archive"

MODEL_PATH_ROOT = os.path.join('ai', 'wave_prediction_model.keras')
SCALER_X_PATH_ROOT = os.path.join('ai', 'scaler_X.pkl')
SCALER_Y_PATH_ROOT = os.path.join('ai', 'scaler_y.pkl')
DB_PATH_ROOT = os.path.join('db', 'user_data.db')


# --- model/scaler utilities ---
@st.cache_resource
def load_model(path=MODEL_PATH_ROOT):
    # load the pre-trained tensorflow model
    if not os.path.exists(path):
        raise FileNotFoundError(f"model file not found at {path}. please run model_trainer.py first.")
    # use compile=False to avoid model loading issues on different tensorflow versions
    return tf.keras.models.load_model(path, compile=False)

@st.cache_resource
def load_scalers():
    # load the data scalers from disk
    if not os.path.exists(SCALER_X_PATH_ROOT) or not os.path.exists(SCALER_Y_PATH_ROOT):
        raise FileNotFoundError("scaler files not found. please run model_trainer.py first.")
    scaler_X = joblib.load(SCALER_X_PATH_ROOT)
    scaler_y = joblib.load(SCALER_Y_PATH_ROOT)
    return scaler_X, scaler_y


# --- heuristic logic (used for data collection and testing) ---

def calculate_heuristic_score(row):
    # calculates a heuristic wave quality score based on swell and wind data.
    height_weight = 0.5
    period_weight = 0.4
    wind_weight = -0.1
    #forecasting model weights
    normalized_height = min(row['swell_wave_height'], 3.0) / 3.0 * 10
    normalized_period = min(row['swell_wave_period'], 15.0) / 15.0 * 10
    normalized_wind = min(row['wind_speed_10m'], 30.0) / 30.0 * 10

    score = (height_weight * normalized_height) + \
            (period_weight * normalized_period) + \
            (wind_weight * normalized_wind)

    # check score bounds
    score = max(1, min(10, score))
    return score


# --- database class (encapsulation & crud) ---

class UserDatabase:
    # handles secure user auth and db ops
    def __init__(self, db_path=DB_PATH_ROOT):
        self._db_path = db_path
        self._initialize_db()

    @property # encapsulation: getter for the db path
    def db_path(self):
        return self._db_path

    def _initialize_db(self):
        # creates the users table if it doesn't exist
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                hashed_password TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def add_user(self, username, password):
        # securely adds a new user with a hashed password (CREATE)
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, hashed_password) VALUES (?, ?)",
                      (username, hashed.decode('utf-8')))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"user {username} already exists.")
            return False
        finally:
            conn.close()

    def verify_user(self, username, password):
        # verifies a user's password against the stored hash (READ)
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("SELECT hashed_password FROM users WHERE username = ?", (username,))
        result = c.fetchone()
        conn.close()

        if result:
            hashed_password = result[0].encode('utf-8')
            return bcrypt.checkpw(password.encode('utf-8'), hashed_password)
        return False

    def modify_user(self, username, new_password):
        # securely updates a user's password (UPDATE/MODIFY)
        new_hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("UPDATE users SET hashed_password = ? WHERE username = ?",
                  (new_hashed.decode('utf-8'), username))
        rows_affected = conn.total_changes
        conn.commit()
        conn.close()
        return rows_affected > 0

    def delete_user(self, username):
        # deletes a user account from the database (DELETE)
        conn = sqlite3.connect(self._db_path)
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE username = ?", (username,))
        rows_affected = conn.total_changes
        conn.commit()
        conn.close()
        return rows_affected > 0

user_db = UserDatabase()


# --- api classes (inheritance and polymorphism) ---

class BaseWeatherAPI:
    # base class for all api fetches. implements polymorphism (fetch_data)
    def __init__(self, latitude, longitude, start_date, end_date):
        self.latitude = latitude
        self.longitude = longitude
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        raise NotImplementedError("subclasses must implement this method")

class OpenMeteoMarineAPI(BaseWeatherAPI):
    # fetches marine weather data (swell and waves)
    def fetch_data(self):
        # polymorphism: implements the base fetch_data method
        url = MARINE_API_URL
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "swell_wave_height,swell_wave_period,wave_direction,sea_level_height_msl",
            "start_date": self.start_date,
            "end_date": self.end_date
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'hourly' in data:
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                return df
        except requests.exceptions.RequestException as e:
            print(f"error during marine api call: {e}")
            return pd.DataFrame()
        return pd.DataFrame()

class OpenMeteoWindAPI(BaseWeatherAPI):
    # fetches wind data (can switch to historical api for past dates)
    def fetch_data(self):
        # polymorphism: implements the base fetch_data method
        is_historical = datetime.strptime(self.start_date, '%Y-%m-%d').date() < datetime.now().date()
        url = HISTORICAL_WEATHER_API_URL if is_historical else WEATHER_API_URL

        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "wind_speed_10m,wind_direction_10m",
            "start_date": self.start_date,
            "end_date": self.end_date
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            if 'hourly' in data:
                df = pd.DataFrame(data['hourly'])
                df['time'] = pd.to_datetime(df['time'])
                return df
        except requests.exceptions.RequestException as e:
            print(f"error during wind api call: {e}")
            return pd.DataFrame()
        return pd.DataFrame()


# --- core functions ---

def geocode_location(location_name):
    # converts a location name to geographical coords (lat/lon)
    try:
        response = requests.get(f"{GEOCODING_API_URL}?name={location_name}")
        response.raise_for_status()
        data = response.json()
        if 'results' in data and data['results']:
            # return the coords of the first result
            return {
                'latitude': data['results'][0]['latitude'],
                'longitude': data['results'][0]['longitude']
            }
    except requests.exceptions.RequestException as e:
        print(f"error during geocoding api call: {e}")
        return None
    return None

def fetch_tide_data(latitude, longitude, start_date, end_date):
    # fetches tide data and finds the next high and low tides
    url = MARINE_API_URL
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "sea_level_height_msl",
        "start_date": start_date,
        "end_date": end_date
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if 'hourly' in data and data['hourly']['sea_level_height_msl']:
            df = pd.DataFrame(data['hourly'])
            df['time'] = pd.to_datetime(df['time'])
            df['sea_level_height_msl'] = df['sea_level_height_msl'].replace(-999, np.nan) # handle missing values

            # use a rolling window to find local minima and maxima
            is_max = df['sea_level_height_msl'] == df['sea_level_height_msl'].rolling(window=3, center=True).max()
            is_min = df['sea_level_height_msl'] == df['sea_level_height_msl'].rolling(window=3, center=True).min()

            high_tides = df[is_max].dropna()
            low_tides = df[is_min].dropna()

            now = datetime.now()
            next_high_tide = high_tides[high_tides['time'] > now].iloc[0] if not high_tides[high_tides['time'] > now].empty else None
            next_low_tide = low_tides[low_tides['time'] > now].iloc[0] if not low_tides[low_tides['time'] > now].empty else None

            result = {}
            if next_high_tide is not None:
                result['next_high_tide'] = {
                    'time': next_high_tide['time'].strftime('%H:%M %p'),
                    'height_m': next_high_tide['sea_level_height_msl']
                }
            if next_low_tide is not None:
                result['next_low_tide'] = {
                    'time': next_low_tide['time'].strftime('%H:%M %p'),
                    'height_m': next_low_tide['sea_level_height_msl']
                }

            return result if result else None

    except requests.exceptions.RequestException as e:
        print(f"error during tide api call: {e}")
    except Exception as e:
        print(f"error processing tide data: {e}")
    return None

def get_surf_forecast_by_name(location_name):
    # fetches the 7-day surf forecast for a given location
    coords = geocode_location(location_name)
    if not coords:
        return pd.DataFrame()

    today = datetime.now().date()
    end_date = today + timedelta(days=6)

    start_date_str = today.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # use the new api classes (polymorphism demo)
    marine_api = OpenMeteoMarineAPI(coords['latitude'], coords['longitude'], start_date_str, end_date_str)
    wind_api = OpenMeteoWindAPI(coords['latitude'], coords['longitude'], start_date_str, end_date_str)

    marine_data = marine_api.fetch_data()
    wind_data = wind_api.fetch_data()

    if not marine_data.empty and not wind_data.empty:
        combined_df = pd.merge(marine_data, wind_data, on='time', how='inner')
        return combined_df
    else:
        return pd.DataFrame()

def predict_surf_quality(data_point):
    # predicts the surf quality score using the trained tensorflow model
    model = load_model()
    scaler_X, scaler_y = load_scalers()

    features = ['swell_wave_height', 'swell_wave_period', 'wind_speed_10m', 'sea_level_height_msl']

    try:
        new_data_df = pd.DataFrame([data_point[features].values], columns=features)

        new_data_scaled = scaler_X.transform(new_data_df)

        # verbose=0 quiets the tensorflow output during prediction
        predicted_scaled = model.predict(new_data_scaled, verbose=0)

        predicted_score = scaler_y.inverse_transform(predicted_scaled)

        return float(predicted_score[0][0])
    except KeyError as e:
        print(f"error: missing feature in data point: {e}. required features are {features}")
        return None
    except Exception as e:
        print(f"error during prediction: {e}")
        return None