import pytest
import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from kookpy import (
    UserDatabase,
    predict_surf_quality,
    calculate_heuristic_score,
    load_model,
    load_scalers
)

@pytest.fixture(scope='module')
def db_test_setup():
    # setup: create a temporary database path for testing
    temp_db_path = 'test_user_data.db'
    db = UserDatabase(db_path=temp_db_path)
    yield db
    # teardown: close connection and delete the temporary db file
    del db
    if os.path.exists(temp_db_path):
        os.remove(temp_db_path)

@pytest.fixture(scope='module')
def sample_data():
    # data point where score should be high (high swell, low wind)
    high_score_data = {
        'swell_wave_height': 2.5,
        'swell_wave_period': 14.0,
        'wind_speed_10m': 5.0,
        'sea_level_height_msl': 0.5
    }
    # data point where score should be low (low swell, high wind)
    low_score_data = {
        'swell_wave_height': 0.1,
        'swell_wave_period': 4.0,
        'wind_speed_10m': 25.0,
        'sea_level_height_msl': 0.0
    }
    return high_score_data, low_score_data

# CRUD Tests
def test_user_add_and_verify(db_test_setup):
    # test user creation and login verification
    db = db_test_setup
    username = "testuser"
    password = "password123"

    # create (add) user
    assert db.add_user(username, password) is True

    # read (verify) user
    assert db.verify_user(username, password) is True
    assert db.verify_user(username, "wrongpass") is False

def test_user_modify_password(db_test_setup):
    # test password update (modify)
    db = db_test_setup
    username = "modifyuser"
    old_pass = "oldpass"
    new_pass = "newpass456"

    db.add_user(username, old_pass)

    # modify user
    assert db.modify_user(username, new_pass) is True

    # verification should fail with old pass and pass with new pass
    assert db.verify_user(username, old_pass) is False
    assert db.verify_user(username, new_pass) is True

def test_user_delete(db_test_setup):
    # test user deletion
    db = db_test_setup
    username = "deleteuser"
    password = "deletepass"

    db.add_user(username, password)

    # delete user
    assert db.delete_user(username) is True

    # user should no longer exist
    assert db.verify_user(username, password) is False

    # attempt to delete again should fail
    assert db.delete_user(username) is False

# Heuristic Logic Test
def test_heuristic_score_boundaries(sample_data):
    # checks that the heuristic score is calculated correctly and is within 1-10
    high_score_data, low_score_data = sample_data

    # check for high score conditions
    high_score = calculate_heuristic_score(high_score_data)
    assert high_score > 7.0
    assert 1.0 <= high_score <= 10.0

    # check for low score conditions
    low_score = calculate_heuristic_score(low_score_data)
    assert low_score < 3.0
    assert 1.0 <= low_score <= 10.0

#AI Model Test
def test_model_prediction_integrity():
    # tests if the trained model can load and predict without error
    # checks if the prediction accuracy (MSE) is within acceptable bounds

    try:
        model = load_model()
        scaler_x, scaler_y = load_scalers()
    except FileNotFoundError:
        pytest.skip("model/scaler files not found. cannot run prediction test.")
        return

    # generate dummy test data (100 synthetic data points)
    n_samples = 100
    features = ['swell_wave_height', 'swell_wave_period', 'wind_speed_10m', 'sea_level_height_msl']

    # create synthetic feature values within realistic ranges
    x_test_raw = pd.DataFrame({
        'swell_wave_height': np.random.uniform(0.1, 3.0, n_samples),
        'swell_wave_period': np.random.uniform(4.0, 15.0, n_samples),
        'wind_speed_10m': np.random.uniform(5.0, 30.0, n_samples),
        'sea_level_height_msl': np.random.uniform(-0.5, 1.0, n_samples),
    })

    # generate synthetic true labels using the heuristic function
    y_test_true = x_test_raw.apply(calculate_heuristic_score, axis=1).values.reshape(-1, 1)

    # scale, predict, and inverse transform the results
    x_test_scaled = scaler_x.transform(x_test_raw)
    y_test_predicted_scaled = model.predict(x_test_scaled, verbose=0)
    y_test_predicted = scaler_y.inverse_transform(y_test_predicted_scaled)

    # calculate mean squared error (mse)
    # mse measures the average squared difference between estimated and actual values.
    mse = mean_squared_error(y_test_true, y_test_predicted)

    # assertion check
    # critical check for the requirement: test the model's performance
    acceptable_mse_threshold = 0.8
    assert mse < acceptable_mse_threshold, f"model mse ({mse:.4f}) is above acceptable threshold of {acceptable_mse_threshold}"