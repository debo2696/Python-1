#%% Importing required libraries
import os.path # To work on the path. Used for relative addressing
import pandas as pd # To read/write excel files
from sklearn.model_selection import train_test_split #To split the data for training and testing
from sklearn.preprocessing import StandardScaler# For data normalization
from sklearn.svm import SVC # SVM
from sklearn.metrics import classification_report,confusion_matrix # Result analysis
#%% Reading '.csv' file
pd.set_option('display.width', 200)#Setting parameters of pandas for display 
pd.set_option(
    'display.float_format',
    lambda x: ('%.7f' % x).rstrip('0').rstrip('.'))#For floating point numbers

df = pd.read_csv(os.path.join(os.path.dirname(__name__), 'iris.csv'))# Reading 'iris.csv' and saving it in dataframe 'df'

species = df[['species']]#Target
params = df.drop('species', axis=1)#Features

#%% Splitting of training and testing data
params_train, params_test, species_train, species_test = train_test_split(
    params,
    species,
    test_size=0.4,# 40% of the total data for testing
    random_state=0)

#%% Data normalization
scaler = StandardScaler()
scaler.fit(params_train)#Normalization parameters (e.g. mean and variance) using training data
params_train_scaled = pd.DataFrame(#training data normalization
    scaler.transform(params_train),
    columns=params_train.columns,
    index=params_train.index)
params_test_scaled = pd.DataFrame(#testing data normalization
    scaler.transform(params_test),
    columns=params_test.columns,
    index=params_test.index)

#%% SVM 
svm= SVC() # Importing library in variable 'svm'
svm.fit(params_train_scaled,species_train)# Training of SVM

pred=svm.predict(params_test_scaled)#Predicting values of test data using trained model
print(confusion_matrix(species_test,pred))# Confusion matrix 


print(classification_report(species_test,pred))#Classification report class wise
print(svm.score(params_test_scaled,species_test)) # Average of classification