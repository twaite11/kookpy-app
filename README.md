# Kookpy AI Surf Forecast

Totally Tubular Wave Predictions Powered by Open-Meteo and TensorFlow

## Project Overview

Kookpy AI Surf Forecast is a full-stack, AI-powered web application designed to predict the quality of surf conditions on a scale of 1 to 10. It utilizes real-time marine and atmospheric data from the Open-Meteo API and processes this information through a trained neural network model.

The application includes robust user authentication (CRUD via SQLite and bcrypt hashing) and a custom retro hacker-style theme.

### Architectural Highlights

Front End: Streamlit (app/app.py) for the user interface.

Core Logic: Centralized in the Python package kookpy/ (includes APIs, CRUD, and heuristic scoring).

AI/Model: Located in the ai/ folder (model_trainer.py, model artifacts).

Database: Local SQLite database (db/user_data.db) for user management.

OOP Principles: Utilizes Inheritance (BaseWeatherAPI), Polymorphism, and Encapsulation (UserDatabase).

## 1. Local Setup and Installation Guide

### Prerequisites

Python 3.9+

git (optional, but recommended for version control)

## Step 1: Fork the Repository (Optional)

git fork https://gitlab.com/wgu-gitlab-environment/student-repos/twaite11/d424-software-engineering-capstone.git

cd kookpy-project


## Step 2: Create and Activate Virtual Environment

python -m venv .venv
# On Windows PowerShell
.\.venv\Scripts\Activate.ps1
# On Linux/macOS
source .venv/bin/activate


## Step 3: Install Dependencies

This project uses a standard setup.py file. The command below installs all required libraries and installs your local kookpy directory as a navigable package using the editable install flag (-e .).

pip install -e .


## Step 4: Run Initial Data Collection and Model Training

The application requires historical data and a trained model before it can run the forecast. These must be run from the project root.

Collect Data (Creates ai/historical_surf_data.csv):

python -m ai.data_collector


Train Model (Creates ai/wave_prediction_model.keras and scalers):

python -m ai.model_trainer


## Step 5: Run the Application

Start the Streamlit web application from the project root.

streamlit run app/app.py


## 2. Maintenance and User Guides

### Maintenance Guide (For Developers)

This guide covers necessary steps for updating, testing, and ensuring integrity.

### Updating Packages

pip install -e . --upgrade

Use this after any changes to setup.py or new dependencies.

### Running Unit Tests

python -m pytest tests/

Run this command from the project root to verify the CRUD and heuristic scoring logic (as defined in tests/test_core_logic.py).

### Retraining the Model

python -m ai.data_collector then python -m ai.model_trainer

Must be done if the data sources or feature engineering logic change.

### Database Access

Use an SQLite browser tool (or run custom Python script) to access db/user_data.db.

DO NOT manually edit hashed_password fields; use the modify_user function for testing.

### User Perspective Guide (For End-Users)

Access: Navigate to the deployed URL.

Authentication: Use the main page form to Sign Up (Username > 3 chars, Password > 5 chars) or Login.

Get Forecast: Once logged in, use the Search by Name tab to find a beach (e.g., "Huntington Beach").

Analyze Results: AI Quality Score: The central metric (1-10) predicting surf quality.

Graphs: Review the 7-day predicted score overlaid on the swell height, wind speed, and tide charts to understand the factors driving the prediction.

Account Management (CRUD): Use the "Manage Account" button in the top right to change your password or permanently delete your user profile.

## 3. Deployment Guide (Streamlit Cloud)

The application is configured for seamless deployment to Streamlit Cloud due to the Python packaging structure.
Create a file named requirements.txt in the project root. This file tells Streamlit's environment manager to install all dependencies and your local package.

### requirements.txt content
pandas
tensorflow
joblib
streamlit
requests
scikit-learn
plotly
numpy
bcrypt
### IMPORTANT: Install the local project as an editable package
-e .


## Push to GitHub

Ensure all files—especially setup.py, requirements.txt, and the entire kookpy/ package—are pushed to a public GitHub repository.

## Deploy to Streamlit Cloud

Go to Streamlit Cloud and log in.

Select New App.

Point the app to your GitHub repository and branch.

Set the Main file path to: app/app.py

Click Deploy! Streamlit will automatically use requirements.txt to install your project correctly.