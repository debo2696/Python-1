#%% Importing required libraries
import os.path # To work on the path. Used for relative addressing
import pandas as pd # To read/write excel files
from sklearn.model_selection import train_test_split #To split the data for training and testing
from sklearn.neural_network import MLPClassifier #Neural Network tool box
from sklearn.preprocessing import StandardScaler# For data normalization

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
    test_size=0.4,# 40% if the total data for testing
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

#%% Neural network 
# 1 hidden layer, same size as the input layer
mlp = MLPClassifier(
#    solver='sgd', # stochastic gradient descent
    solver='lbfgs', # optimizer in the family of quasi-Newton methods.
    hidden_layer_sizes=(params.shape[1],),# No. of neurons in hidden layer = no. of inputs
    random_state=0)
mlp.fit(params_train_scaled, species_train.species)# Training a model

print 'Train score: %.3g' % mlp.score(params_train_scaled, species_train)# Training accuracy
print 'Test Score: %.3g' % mlp.score(params_test_scaled, species_test)# Testing accuracy
#print

#%% Analysis of results
#Predicted classes
predicted = pd.DataFrame(
    mlp.predict(params_test_scaled),
    columns=['predicted'],
    index=params_test_scaled.index)
actual = species_test.rename(columns={'species': 'actual'})#Actual classes
concat = pd.concat((actual, predicted), axis=1)#Concatenating Actual and Predicted classes

#Failed classification cases
failed = params_test.join(
    concat[concat.predicted != concat.actual],
    how='inner')

#Printing failed classification cases with features, neural out
if failed.empty:
    print 'Classified all test samples correctly!'
    print

else:
    failed_params_scaled = scaler.transform(
        failed.drop('actual', axis=1).drop('predicted', axis=1))
    failed_with_proba = pd.DataFrame(
        mlp.predict_proba(failed_params_scaled),
        columns=mlp.classes_,
        index=failed.index)

    print 'Failed to classify the following test samples:'
    print pd.concat((failed, failed_with_proba), axis=1)
    print

# display the network weights for each layer rows in each table represent a
# neuron, and columns represent the input from the previous layer, a cell
# gives us the weight to use for a given neuron + input value
print 'Network weights, w/o bias, input layer is layer #0:'
for i, weight_matrix in enumerate(mlp.coefs_, start=1):
    print 'Layer #%d' % i

    if i == 1:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns = df.drop('species', axis=1).columns

    else:
        weight_matrix_df = pd.DataFrame(weight_matrix.transpose())
        weight_matrix_df.columns = ['Neuron #%d from layer #%d' % (c, i - 1)
                                    for c in weight_matrix_df.columns]

    weight_matrix_df.index = weight_matrix_df.index.map(
        lambda x: 'Neuron #%d' % x)
    print weight_matrix_df

    print
