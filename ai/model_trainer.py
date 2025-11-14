import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
import joblib
import os
import kookpy

def build_and_train_model(x_train, y_train, epochs=100):
    # bread and butter of creating the actual model

    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu',
                           input_shape=(x_train.shape[1],)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    print("starting model training...")
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, verbose=1)
    print("model training complete.")
    return model


def save_model_and_scalers(model, scaler_x, scaler_y,
                           model_path='wave_prediction_model.keras',
                           scaler_x_path='scaler_X.pkl',
                           scaler_y_path='scaler_y.pkl'):
    # save the model and scalers to the ai/ directory relative to the project root
    base_dir = 'ai'
    model.save(os.path.join(base_dir, model_path))
    joblib.dump(scaler_x, os.path.join(base_dir, scaler_x_path))
    joblib.dump(scaler_y, os.path.join(base_dir, scaler_y_path))
    print("\nmodel and scalers saved successfully.")


if __name__ == '__main__':

    # file path relative to the project root
    file_path = os.path.join('ai', 'historical_surf_data.csv')

    if not os.path.exists(file_path):
        print(f"error: data file not found at '{file_path}'.")
        print("please run 'data_collector.py' first to generate the historical data.")
    else:
        # load and drop if missing
        df = pd.read_csv(file_path, parse_dates=['time'])
        df.dropna(inplace=True)

        if df.empty:
            print(
                "error: empty dataframe after dropping n/a rows.")
        else:
            # define features and target
            features = ['swell_wave_height', 'swell_wave_period',
                        'wind_speed_10m', 'sea_level_height_msl']
            target = 'wave_quality_score'

            # check if all required columns exist
            if not all(col in df.columns for col in features + [target]):
                print("error missing column in data file.")
                print(f"required columns: {features + [target]}")
            else:
                x = df[features]
                y = df[target]

                # split the data into training and testing sets
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=42)

                # scale the features and target data
                scaler_x = StandardScaler()
                x_train_scaled = scaler_x.fit_transform(x_train)

                scaler_y = StandardScaler()
                y_train_scaled = scaler_y.fit_transform(
                    y_train.values.reshape(-1, 1))

                # build and train the model
                model = build_and_train_model(x_train_scaled, y_train_scaled)

                # save the model and scalers
                save_model_and_scalers(model, scaler_x, scaler_y)