# Housing Price Analysis

A research project inspired by Yandex ML trainings. It's for predicting **house prices** in Ames with **exploratory analysis (EDA)** and **visual comparisons** of regression models.

The project includes:
- **Preprocess continuous and categorical features**
- **Custom regression models**
- **Оne-hot encoding** for categorical variables
- **Training pipelines**
- **Visualize data and model performance** via metrics such as MAE, RMSLE, etc

---

## Table of Contents

- [Housing Price Analysis](#housing-price-analysis)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Data description](#data-description)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Where is the project?...](#where-is-the-project)
    - [Scripts](#scripts)
  - [Classes and Models](#classes-and-models)
    - [Data Preprocessing](#data-preprocessing)
    - [Custom Models](#custom-models)
    - [Pipelines](#pipelines)
  - [📊 Visualizations](#-visualizations)
    - [Data Analysis](#data-analysis)
    - [Feature Correlations](#feature-correlations)
    - [Location Analysis](#location-analysis)
    - [Model Performance](#model-performance)
    - [Training Dynamics](#training-dynamics)
    - [Prediction Accuracy](#prediction-accuracy)
  - [Results](#results)
---

## Overview

This project builds a regression model to predict housing prices in [Ames, Iowa](https://www.openml.org/d/41211).  
I use:

- **Continuous data transformations**: scaling.
- **Categorical data encoding** via `OneHotEncoder`.
- **Custom linear regression** using stochastic gradient descent, with optional L2 regularization.
- **Hyperparameter tuning**: grid search and cross-validation.
- **Logarithmic transformations** of the target variable for improved model stability.

---

## Data description

The data is split into **training** and **testing** sets.
The **Ames Housing** key columns include:

1. **Continuous features**: `Year_Built`.
2. **Categorical features**: `Overall_Qual`.
3. **Target variable**: `Sale_Price`.


---

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/estnafinema0/Housing-Price-Analysis.git
   cd Housing-Price-Analysis
   ```
2. **Create a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate   
   ```
   Working on Linux.
3. **Install requirements**:

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Where is the project?...

The complete analysis is in `notebook.ipynb`.
Just open and run cells in sequence! 

### Scripts

If you want to check particular parts of the project, look at the `scripts/` folder.

## Classes and Models

### Data Preprocessing

1. **`BaseDataPreprocessor`**  
   - Picks out number-based columns you want to use
   - Makes all numbers work on the same scale

2. **`SmartDataPreprocessor`**  
   - Adds helpful new data like distance to city center
   - Fixes missing data by using middle values
   - Makes numbers ready for the model to use
   - Makes predictions better by using real-world knowledge

3. **`OneHotPreprocessor`**  
   - Built on top of `BaseDataPreprocessor`
   - Turns text data (like house zones and sale types) into numbers the model can use

### Custom Models

1. **`ExponentialLinearRegression`**  
   - A special version of the `Ridge` model
   - Changes house prices to a better format while learning
   - Changes them back when making predictions

2. **`SGDLinearRegressor`**  
   - A model that learns step by step
   - Keeps track of how well it's learning
   - Shows you how it improves over time

### Pipelines

We have several ready-to-use pipelines to make predictions:

1. **`make_base_pipeline()`**
   - The simple version that works with just numbers
   - Uses `BaseDataPreprocessor` and basic `Ridge` model

2. **`make_onehot_pipeline()`**
   - Our best performer!
   - Handles both numbers and categories (like house zones)
   - Uses `OneHotPreprocessor` to turn text into numbers

3. **`make_smart_pipeline()`**
   - Uses `SmartDataPreprocessor` to add helpful new data
   - Good for when you want to use location data

Each pipeline combines data preparation and model training into one easy step. Just use `fit()` and `predict()`! 😊

---

## 📊 Visualizations

Step-by-step visualizations.

### Data Analysis

![Price Distribution](/visualizations/Distribution%20of%20Sale%20Price.png)
The distribution of house prices shows a clear right-skew pattern. We found that log-transformation makes the data more normally distributed, which helps our models perform better.

### Feature Correlations

![Feature Correlations](/visualizations/Feature%20Correlations.png)

For 'Sale_Price' could be important location-based features (Longitude, Latitude). Also they show weak correlations with other features.

### Location Analysis

![Price vs. Distance to Center](/visualizations/Price%20vs.%20Distance%20to%20Center.png)
![Property Locations Colored by Price](/visualizations/Property%20Locations%20Colored%20by%20Price.png)
We see interesting neighborhood patterns:
- Higher-priced clusters in northern areas
- Price variations more tied to neighborhood than distance to center

### Model Performance

![Model Comparison](/visualizations/Model%20Comparison.png)
Our model comparison shows:
- OneHot Pipeline leading with lowest MAE (~18,000)
- Clear performance ranking: OneHot > Exponential > Base > SGD

### Training Dynamics

![SGD Training](visualizations/SGD%20Dynamics.png)
The SGD Regressor's training shows:
- Loss stabilization around 800 iterations
- After 800 iterations, the model's performance starts to decrease, because of the overfitting.

### Prediction Accuracy

![Predictions vs Actual](visualizations/Scatter%20plots:%20predicted%20vs.%20actual.png)
The scatter plots show that the OneHot Pipeline follows the ideal prediction line most closely.

> 💡 Check out `notebook.ipynb` to recreate these visualizations.
---

## Results

| Model                    | MAE    | RMSLE  | Notes                                          |
|-------------------------|--------|---------|------------------------------------------------|
| **OneHot Pipeline**     | 18,000 | 0.155  | Best performer! Great with categorical features |
| **Base Pipeline**       | 23,000 | 0.190  | Simple but stable baseline                     |
| **Exponential Pipeline**| 20,500 | 0.182  | Good with price distribution                   |
| **SGD Regressor**       | 26,000 | 0.200  | Shows instability after 800 iterations         |

> 💡 **Key results:**
> - OneHot Pipeline shows best results across both metrics
> - SGD Regressor needs more tuning to compete with other models
---

Thanks for checking out, guys! 
