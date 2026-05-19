# Airline Passenger Satisfaction: Exploratory Data Analysis and Classification Modeling

## Problem Statement

The Airline Passenger Satisfaction dataset contains survey responses from 103,904 airline passengers, collected across a variety of flight experiences. For each passenger, the dataset includes demographic information, flight details, and ratings for 14 service categories, as well as a binary satisfaction label: satisfied or neutral/dissatisfied.

The goal of this project is to analyze the dataset, clean and prepare the data, and build a classification model capable of predicting whether a passenger will be satisfied or not. This analysis is useful for understanding which factors most influence passenger satisfaction and for building a model that airlines could use to identify at-risk experiences.

## Data Dictionary

| Column Name | Description |
|---|---|
| id | Unique passenger identifier |
| gender | Passenger gender (Male / Female) |
| customer_type | Whether the passenger is a loyal or disloyal customer |
| age | Age of the passenger |
| type_of_travel | Purpose of the flight (Business / Personal) |
| class | Travel class (Business / Eco / Eco Plus) |
| flight_distance | Distance of the flight in miles |
| inflight_wifi_service | Rating of inflight Wi-Fi service (0–5) |
| departure/arrival_time_convenient | Rating of departure/arrival time convenience (0–5) |
| ease_of_online_booking | Rating of the online booking experience (0–5) |
| gate_location | Rating of gate location convenience (0–5) |
| food_and_drink | Rating of food and drink quality (0–5) |
| online_boarding | Rating of the online boarding process (0–5) |
| seat_comfort | Rating of seat comfort (0–5) |
| inflight_entertainment | Rating of inflight entertainment (0–5) |
| on-board_service | Rating of on-board service (0–5) |
| leg_room_service | Rating of leg room service (0–5) |
| baggage_handling | Rating of baggage handling (0–5) |
| checkin_service | Rating of check-in service (0–5) |
| inflight_service | Rating of inflight service (0–5) |
| cleanliness | Rating of cabin cleanliness (0–5) |
| departure_delay_in_minutes | Departure delay in minutes |
| arrival_delay_in_minutes | Arrival delay in minutes |
| satisfaction | Target variable: satisfied or neutral or dissatisfied |

## Executive Summary

This project was divided into three main stages: data cleaning and exploratory data analysis (EDA), classification modeling, and an interactive Streamlit application.

During the first stage, the dataset was cleaned and prepared for modeling. The 5 categorical columns (gender, customer_type, type_of_travel, class, and satisfaction) were label-encoded into numerical values. The only column with missing values was `arrival_delay_in_minutes`, which had 310 null entries (0.3% of the dataset). Since the distribution of this column was heavily right-skewed, missing values were replaced with the median. Outliers in the delay columns were identified through box plots but were kept in the dataset, as extreme delays represent real events rather than data entry errors.

In the EDA phase, the relationships between variables and passenger satisfaction were explored using correlation analysis, histograms, box plots, and scatter plots. The strongest positive correlations with satisfaction were found in online boarding, travel class, type of travel, and inflight entertainment.

In the modeling stage, two classification models were trained and evaluated: Logistic Regression as a baseline and Random Forest as a more powerful alternative. The data was split into 80% training and 20% testing sets with stratification, and all features were scaled using StandardScaler. Random Forest significantly outperformed Logistic Regression in all metrics.

Finally, an interactive Streamlit app was built that allows users to explore the data, train and compare models, make individual predictions, and view feature importance and correlation insights.

### Data Cleaning Steps

- Removed identifier columns (`Unnamed: 0` and `id`) that provide no predictive value.
- Standardized column names to lowercase with underscores.
- Label-encoded 5 categorical columns into numerical values using `.map()`:
  - Binary columns (gender, customer_type, type_of_travel, satisfaction) → 0 and 1.
  - Three-category column (class) → 0, 1, 2.
- Identified 310 missing values in `arrival_delay_in_minutes` (0.3% of data). Replaced with the median due to the column's right-skewed distribution.
- Detected extreme outliers in both delay columns (up to ~1,600 minutes). Kept them in the dataset as they represent real flight delays, not measurement errors.
- Confirmed no duplicate rows in the dataset.
- Exported the cleaned dataset as `cleaned_airline_passenger_satisfaction.csv` for use in modeling.

### Key Visualizations

#### Visualization 1: Correlation with Satisfaction
A heatmap was used to rank all variables by their correlation with the satisfaction column. The four strongest positive correlations were:
- Online boarding
- Class
- Type of travel
- Inflight entertainment

#### Visualization 2: Distribution of Key Features
Histograms were used to analyze the distribution of numerical variables. Both delay columns showed heavily right-skewed distributions with most passengers experiencing little to no delay, and a small number of extreme outliers creating long tails.

#### Visualization 3: Box Plots — Outlier Detection
Box plots were generated for the departure and arrival delay columns to visualize the outlier distribution. The interquartile range for both columns was concentrated close to 0, confirming that the vast majority of flights had minimal delays.

## Model Performance

### Model Selection
Two classification models were trained and evaluated:

- **Logistic Regression** was selected as the baseline model due to its simplicity and interpretability.
- **Random Forest Classifier** was selected as the improved model due to its ability to capture non-linear relationships and interactions between features.

Both models were trained using StandardScaler-normalized features and evaluated against a baseline accuracy of 56.67% (the majority class proportion).

### Evaluation Metrics

| Model | Training R² | Test R² |
|---|---|---|
| Baseline (majority class) | — | 56.67% |
| Logistic Regression | 87.44% | 87.70% |
| Random Forest | 100% | 96.26% |

### Interpretation

Both models significantly outperformed the baseline. Logistic Regression achieved consistent performance between training and testing (~87%), suggesting good generalization with no signs of overfitting.

Random Forest achieved near-perfect training accuracy (100%) and 96.26% on the test set. The test performance is strong and represents a meaningful improvement over Logistic Regression.

In terms of classification errors, Random Forest reduced total errors from 2,557 (Logistic Regression) to 778 — a reduction of over 69%. Specifically, Type I errors dropped from 1,132 to 275 and Type II errors from 1,425 to 503.

## Conclusions and Recommendations

- Online boarding, inflight entertainment, travel class, and type of travel are the strongest predictors of passenger satisfaction.
- Random Forest is clearly the better model for this problem, achieving over 96% accuracy on unseen data.
- Airlines should prioritize improving the digital experience (online boarding, ease of online booking) and in-flight service quality (entertainment, seat comfort), as these appear consistently in both correlation and feature importance analyses realized in the Streamlit part.
- Loyal business travelers in higher travel classes tend to report higher satisfaction — airlines could focus retention efforts on converting casual travelers into loyal customers.

## Additional Information

The dataset contains survey and operational data from airline passengers and is commonly used for classification modeling practice. The interactive Streamlit application (`App_Airlines.py`) allows users to explore the data, compare model performance, and make real-time predictions by adjusting passenger input values.