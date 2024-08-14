# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 16:30:06 2024

@author: MALATHI PALADUGU
"""
#For doing the question 4 I have followed the links mentioned below
#https://www.youtube.com/watch?v=5XnHlluw-Eo&t=252s
#https://chat.openai.com/c/03a2b147-2e4b-41eb-8f6c-da4665631231
#https://www.youtube.com/watch?v=WLwjvWq0GWA
#The RF regression model is applied on the red wine data set and set the sliders for different features in the data set. 
#Varying the proportion of features will change the mean square error 
#The sliders are used to vary the feature value. The error varies accordingly
#Deployed this model on a local web server using streamlit app.


#Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import streamlit as st

#Downloading the redwine data from the website
dataset_redwine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv', sep = ';')#reading the redwine csv file into data_readwine from the csv files uploaded in the folder
#Quality has been taken as the target vector, so dropping it from the data set to sepearate data and target
data_redwine = dataset_redwine.drop(columns=['quality'])#seperating the features and target
y_red = dataset_redwine['quality']#Target vector

#splitting the data into training and testing with the ratio 80:20
data_redwine_train, data_redwine_test, y_red_train, y_red_test = train_test_split(data_redwine, y_red, test_size = 0.2, random_state=42)

#Finding the correlation coeffcients
corr_mat_red = data_redwine_train.corr(method = 'spearman')#Spearman used for assesing non-linear relationship among the variable
#For redwine
mod_corr_mat_red = corr_mat_red.copy()

#If the corr coeff <0.8 it is made as 0 and found out the features with 
#Corr oeff >0.8 using the below code. It removes the correlated features and final features are stored in cols_to_retain
mod_corr_mat_red[abs(mod_corr_mat_red) < 0.8] = 0
corr_mat_red_array = mod_corr_mat_red.to_numpy()
indices = np.where(abs(corr_mat_red_array)>0.8)

rows_red = indices[0]
cols_red = indices[1]
row_head_red = corr_mat_red.index.tolist()
col_head_red = corr_mat_red.index.tolist()
pairs_red = [(row_head_red[row], col_head_red[col]) for row, col in zip(rows_red, cols_red)]
#Remove the duplicate pairs
unique_pairs_red = set()

for pair in pairs_red:
  unique_pairs_red.add(tuple(sorted(pair)))
for pair in unique_pairs_red:
  header_value = pair[0]
  for header in pair:
    corr_mat_red.rename(columns = {header: header_value}, index = {header: header_value}, inplace = True)
row_head_red = corr_mat_red.index.tolist()
col_head_red = corr_mat_red.columns.tolist()

row_headers_unique_red = list(set(row_head_red))
#redwine_col_to_retain = row_headers_unique_red.columns.tolist()
redwine_col_to_retain = col_head_red

#Preprocessing using standard scaler(normalising) the data
scaler_red = preprocessing.StandardScaler()#Normalises the data for red is named as  scale_coeff_red


scaler_red.fit(data_redwine_train[redwine_col_to_retain])#It computes mean and standard deviation for redwine training data set

#Training data of input and target after normalisation
train_X_red = pd.DataFrame(scaler_red.transform(data_redwine_train[redwine_col_to_retain]), columns = redwine_col_to_retain)#Uses mean and std deviation
train_Y_red = pd.DataFrame(y_red_train)

#Testing on redwine data
test_X_red = pd.DataFrame(scaler_red.transform(data_redwine_test[redwine_col_to_retain]),columns = redwine_col_to_retain)
test_Y_red = pd.DataFrame(y_red_test)

##***************************************#
##Modeling the red wine data using RF##
##**************************************#
#Even though the target variable is integer
#Regression has been used in this question as it is mentioned to do so
scoring = 'neg_mean_squared_error'#scoring parameter is taken as mean square error
rf_model = RandomForestRegressor()
hp = {'max_depth': [2,5,10,20],'n_estimators': [10,20,30,50,100]}

#Using grid search for the data which is splitted as training
grid_search_rf_red = GridSearchCV(estimator = rf_model, param_grid = hp, cv =5, scoring = scoring)
grid_search_rf_red.fit(train_X_red, train_Y_red.values.ravel())#Fitting the model



#Finding the best parameters for white wine data
best_params_red = grid_search_rf_red.best_params_#best hyperparameters can be found
best_score_red = grid_search_rf_red.best_score_
best_rf_model_red = grid_search_rf_red.best_estimator_#Finds the best model
rf_model_red = RandomForestRegressor(max_depth = best_params_red['max_depth'], n_estimators = best_params_red['n_estimators'])  # calling the model and passing optimum hyperparameter
rf_model_red.fit(train_X_red, train_Y_red.values.ravel())
y_pred_red = best_rf_model_red .predict(data_redwine_test)#Predicting the output for the test input



st.title('Wine Quality Predictor')

#Giving slider for each feature
fixed_acidity = st.slider('fixed acidity', float(data_redwine['fixed acidity'].min()), float(data_redwine['fixed acidity'].max()), float(data_redwine['fixed acidity'].mean()))
volatile_acidity = st.slider('volatile acidity', float(data_redwine['volatile acidity'].min()), float(data_redwine['volatile acidity'].max()), float(data_redwine['volatile acidity'].mean()))
citric_acid = st.slider('citric acid', float(data_redwine['citric acid'].min()), float(data_redwine['citric acid'].max()), float(data_redwine['citric acid'].mean()))
residual_sugar = st.slider('residual sugar', float(data_redwine['residual sugar'].min()), float(data_redwine['residual sugar'].max()), float(data_redwine['residual sugar'].mean()))
chlorides = st.slider('chlorides', float(data_redwine['chlorides'].min()), float(data_redwine['chlorides'].max()), float(data_redwine['chlorides'].mean()))
free_sulfur_dioxide = st.slider('free sulfur dioxide', float(data_redwine['free sulfur dioxide'].min()), float(data_redwine['free sulfur dioxide'].max()), float(data_redwine['free sulfur dioxide'].mean()))
total_sulfur_dioxide = st.slider('total sulfur dioxide', float(data_redwine['total sulfur dioxide'].min()), float(data_redwine['total sulfur dioxide'].max()), float(data_redwine['total sulfur dioxide'].mean()))
density = st.slider('density', float(data_redwine['density'].min()), float(data_redwine['density'].max()), float(data_redwine['density'].mean()))
pH = st.slider('pH', float(data_redwine['pH'].min()), float(data_redwine['pH'].max()), float(data_redwine['pH'].mean()))
sulphates = st.slider('sulphates', float(data_redwine['sulphates'].min()), float(data_redwine['sulphates'].max()), float(data_redwine['sulphates'].mean()))
alcohol = st.slider('alcohol', float(data_redwine['alcohol'].min()), float(data_redwine['alcohol'].max()), float(data_redwine['alcohol'].mean()))

#All the features are saved into it
Features = pd.DataFrame({'fixed acidity': [fixed_acidity], 'volatile acidity': [volatile_acidity], 'citric acid': [citric_acid], 
                         'residual sugar': [residual_sugar], 'chlorides': [chlorides], 'free sulfur dioxide': [free_sulfur_dioxide],
                         'total sulfur dioxide': [total_sulfur_dioxide], 'density': [density], 'pH': [pH], 'sulphates': [sulphates], 
                         'alcohol': [alcohol]})
prediction = rf_model_red.predict(Features)#Value of the feature is taken
mse = mean_squared_error(test_Y_red,y_pred_red )#Mean square error is computed


st.subheader('Predicted Wine Quality')

st.write(prediction)
st.subheader('Mean square error with the given parameters')
st.write(mse)
