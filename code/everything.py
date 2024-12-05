# File to see everything better
#################################

# packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# load data #############################
df = pd.read_excel('food_affordability.xls', sheet_name='Food_afford_cdp_co_region_ca')
df = df.drop(df.index[-1]) # remove last row because it is not part of the data

# make factors ############################
df['region_code'] = df['region_code'].astype('category')
df['race_eth_code'] = df['race_eth_code'].astype('category')
df['geotypevalue'] = df['geotypevalue'].astype('category')
df['county_fips'] = df['county_fips'].astype('category')

# Compute Diversity Index - this section currently is not working ###########################
# using Simpson's Diversity Index
# measures the diversity of a dataset in the context of demographic studies
race_counts = df.groupby(['geoname', 'race_eth_name']).size().unstack(fill_value=0)
race_proportions = race_counts.div(race_counts.sum(axis=1), axis=0)
# Compute Simpson's Diversity Index
df['diversity_index'] = 1 - (race_proportions**2).sum(axis=1)
df['diversity_index'].notnull().sum()

# add lat/lon features #############################
all_zipcodes = pd.read_csv('zip_code_database.csv')
# names of all counties in the dataset
counties = ['Alameda', 'Alpine', 'Amador', 'Butte', 'Calaveras', 'Colusa',
       'Contra Costa', 'Del Norte', 'El Dorado', 'Fresno', 'Glenn',
       'Humboldt', 'Imperial', 'Inyo', 'Kern', 'Kings', 'Lake', 'Lassen',
       'Los Angeles', 'Madera', 'Marin', 'Mariposa', 'Mendocino',
       'Merced', 'Modoc', 'Mono', 'Monterey', 'Napa', 'Nevada', 'Orange',
       'Placer', 'Plumas', 'Riverside', 'Sacramento', 'San Benito',
       'San Bernardino', 'San Diego', 'San Francisco', 'San Joaquin',
       'San Luis Obispo', 'San Mateo', 'Santa Barbara', 'Santa Clara',
       'Santa Cruz', 'Shasta', 'Sierra', 'Siskiyou', 'Solano', 'Sonoma',
       'Stanislaus', 'Sutter', 'Tehama', 'Trinity', 'Tulare', 'Tuolumne',
       'Ventura', 'Yolo', 'Yuba']
# match my counties with those in the zipcode database
city_zipcodes = all_zipcodes[all_zipcodes['primary_city'].isin(counties)]
features_to_join = city_zipcodes[['latitude', 'longitude', 'primary_city']]
features_to_join['county_name'] = features_to_join['primary_city']
features_to_join = features_to_join.drop('primary_city', axis = 1)
# merge datasets
df = pd.merge(df, features_to_join, on = 'county_name', how = 'outer')

# get all the numeric/categorical features together #########################
numeric_predictors = df.select_dtypes(include=['number']).drop('affordability_ratio', axis = 1) # df
numeric_predictors = numeric_predictors.drop(numeric_predictors.filter(regex='^is').columns, axis=1)
numeric_predictors = numeric_predictors.drop(columns = 'CA_RR_Affordability')
num_pred_names = numeric_predictors.columns

categorical_predictors = df.select_dtypes(include=['object', 'category']) # df
dummy_vars = df.filter(regex='^is')  # assuming dummy variables start with "is"
categorical_predictors = pd.concat([categorical_predictors, dummy_vars], axis=1)
cat_pred_names = categorical_predictors.columns
# consistent types for numeric columns
df[num_pred_names] = df[num_pred_names].apply(pd.to_numeric, errors='coerce')
# consistent types for categorical columns
df[cat_pred_names] = df[cat_pred_names].astype(str)

# household-level affordability ratio #################################
df['affordability_per_person'] = df['affordability_ratio'] / df['ave_fam_size']

# save to csv #############################
df.to_csv('final_data.csv', index = False)

# Supervised Learning Models ###################################
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost.sklearn import XGBRegressor

# Data Prep ########################################
df['median_income'] = df['median_income'].fillna(df['median_income'].mean()) # make sure response has no NAs
# get my X and Y matrices
X = df.drop(columns = ['median_income', 'version', 'ind_id', 'ind_definition', 'reportyear'])
y = df['median_income']
# extract only the column names for each type
numeric_features = X.select_dtypes(include = ['number']).columns
categorical_features = X.select_dtypes(include = ['object']).columns

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=307) # split the data
# make transformers for each data type
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  
    ('poly', PolynomialFeatures(include_bias=False)),
    ('scaler', StandardScaler())                  
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ('filtering', SelectPercentile(f_regression, percentile=50))     
])
# run both through the preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),      
        ('cat', categorical_transformer, categorical_features)  
    ])

# may need to use samples for testing out some models
X_train_sample = X_train.sample(n=50000, random_state=42)
y_train_sample = y_train.loc[X_train_sample.index]
# scaling response - do this to full train/test when I am done with the samples
scaler = StandardScaler()
y_train_scaled = scaler.fit_transform(y_train_sample.values.reshape(-1, 1))
y_test_scaled = scaler.transform(y_test.values.reshape(-1, 1))
# taking the log of income may make most sense
y_train_log = np.log1p(y_train_sample)
y_test_log = np.log1p(y_test)

# KNN Regressor ############################
# sample KNN Regressor
pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', KNeighborsRegressor(n_neighbors=5, weights='distance'))  
])
pipe.fit(X_train_sample, y_train_log)
y_test_pred = pipe.predict(X_test)
test_mse = mean_squared_error(y_test_log, y_test_pred)
print(test_mse) #0.020348175708673553

# Linear Regression ###############################
# sample Linear Regression
lr_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())  
])
lr_pipe.fit(X_train_sample, y_train_sample)
y_test_pred = pipe.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(test_mse) # 0.16282604899371717 scaled

# evaluate model using cross-validation
scores = cross_val_score(lr_pipe, X_train_sample, y_train_scaled, scoring='neg_mean_squared_error', cv=5)
print(f"Average MSE: {-scores.mean()}")

# now test the model using grid search with ridge
ridge_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Ridge())
])
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]  # Regularization strength for Ridge
}

# GridSearchCV
grid_search = GridSearchCV(ridge_pipe, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_sample, y_train_sample)

# Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation MSE: {-grid_search.best_score_}")
#Best Parameters: {'model__alpha': 0.1}
#Best Cross-Validation MSE: 40669920.493706405

# now test the model using grid search with lasso - takes a lot longer than Ridge
lasso_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', Lasso())
])
param_grid = {
    'model__alpha': [0.01, 0.1, 1, 10, 100]  # Regularization strength for Lasso
}

# GridSearchCV
grid_search = GridSearchCV(lasso_pipe, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_sample, y_train_sample)
# Results
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation MSE: {-grid_search.best_score_}")

# Random Forest #############################
# make sure I don't have any nulls
X_train_sample.isnull().sum()
# since I do, I need to impute missing values
# make new numeric transformer since I only want to impute now
dt_numeric_transformer = Pipeline(steps = [
    ('imputer', SimpleImputer(strategy='mean'))
])
dt_preprocessor = ColumnTransformer(
    transformers=[
        ('num', dt_numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)
pipe = Pipeline(steps=[
    ('preprocessor', dt_preprocessor),
    ('model', DecisionTreeRegressor(random_state = 42))  
])
pipe.fit(X_train_sample, y_train_sample) # do not use scaled versions because decision trees don't need that
y_test_pred = pipe.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
print(test_mse)

# tuning my decision tree
# using grid search
param_grid = {
    'model__max_depth': [3, 5, 10, None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4]
}
grid_search = GridSearchCV(pipe, param_grid, scoring='neg_mean_squared_error', cv=5)
grid_search.fit(X_train_sample, y_train_sample)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best MSE: {-grid_search.best_score_}")

# Gradient Boost - not working #################################
# using XGBoost
xgb_model = XGBRegressor(n_estimators=100, 
                         learning_rate=0.1, 
                         max_depth=3, 
                         random_state=42,
                        )
xgb_model.fit(X_train_sample, y_train_sample)
# maybe add l1/l2 regularization 

# evaluate model
y_test_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_test_pred)
print(f'Test MSE: {mse}')
# graph feature importance
feature_importances = xgb_model.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.show()
# tune hyperparams with grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 1.0]
}

gb_grid_search = GridSearchCV(
    XGBRegressor(random_state=42),
    param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

gb_grid_search.fit(X_train_sample, y_train_sample)
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best MSE: {-grid_search.best_score_}")
# graph feature importance
feature_importances = xgb_model.feature_importances_
plt.barh(range(len(feature_importances)), feature_importances)
plt.xlabel('Feature Importance')
plt.ylabel('Feature Index')
plt.show()

# Deep Learning #################################
# load all DNN packages
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from keras.layers import LeakyReLU
import tensorflow as tf