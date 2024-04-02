# Title: Laptop Price Prediction Web Application

# Introduction:
This Python script develops a web application for predicting laptop prices based on various features such as company, RAM, weight, touchscreen availability, display type, storage type, CPU, GPU, operating system, etc. The prediction model is built using a Random Forest Regressor and deployed using Flask, a web framework.

# Code Overview:
- Data Cleaning:
  The dataset is read from 'laptop_data.csv'.
  Data cleaning steps include handling missing values, converting data types, extracting features, and encoding categorical variables.
- Feature Engineering:
  Features such as RAM, weight, touchscreen, IPS, display resolution, storage type, GPU brand, and operating system are extracted or engineered from existing features.
- Model Training:
  Random Forest Regressor is chosen as the prediction model.
  Data is split into training and testing sets.
  One-hot encoding is applied to categorical features using sklearn's ColumnTransformer and Pipeline.
- Model Evaluation:
  The trained model's performance is evaluated using the R^2 score on the test set.
- Flask Web Application:
  Flask routes are defined to handle user requests.
  An HTML form is created for users to input laptop specifications.
  User inputs are used to make price predictions using the trained model.

# Dependencies:
- NumPy
- Pandas
- scikit-learn
- Flask

# Instructions:
- Ensure the 'laptop_data.csv' file containing laptop data is in the specified directory.
- Install the required dependencies if not already installed.
- Run the script to start the Flask web application.
- Access the web interface via a web browser and input laptop specifications to obtain price predictions.

# Author:
Prajesh Tejani

# References:
- scikit-learn documentation: https://scikit-learn.org/
- Flask documentation: https://flask.palletsprojects.com/






