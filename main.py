#detecting cc fraud through an unbalanced data set(which is
#not great to train a model because it is one class to heavy)
#the use of linear regression, and then some other type, and then some other type

import numpy as np #allows mathmatical functions for multi-dimensional arrays
import pandas as pd #allows for data frames 
from sklearn.model_selection import train_test_split #allows for training and testing data
from sklearn.linear_model import LogisticRegression #allows for predicition classification of either the pos or neg class
from sklearn.metrics import accuracy_score #allows determination of how accurate the model is

def logistic_regression():
    data = pd.read_csv('/Users/qusayqadir/Documents/code/CC-Fraud/creditcard.csv') #reads the csv data 
    print(data)
    print(data['Class'].value_counts()) #shows the unbalanced data set, where 0 represents legit transaction and 1 represents fraud
    f_transactions = data[data.Class == 1] ##known fradulent transactions from data set 
    print(f_transactions) ##fraduelent transactions
    r_transactions = data[data.Class == 0] ##
    print(r_transactions) ##real transactions
    r_transactions_data = r_transactions.Amount.describe()
    print(r_transactions_data)
    f_transactions_data = f_transactions.Amount.describe()
    print(f_transactions_data)
    #compare values between r_transactions and f_transactions.
    comparing = data.groupby('Class').mean()
    print(comparing)
    #since the dataset is imbalanced, data['Class'].value_counts() --> take equal number of random samples from r_transactions that are there in f_transactions
    r_samples = r_transactions.sample(n=492)
    combined_sample = pd.concat([r_samples, f_transactions], axis = 0) ##if axis is 0 data is added row wise if axis = 1 then data is added column wise
    print(combined_sample['Class'].value_counts()) ##shows equal data set
    combined_sample.groupby('Class').mean() #shows that data set is a good sample
    
    new_data = combined_sample.drop(columns='Class', axis=1) #loc method will be used to set a condition 
    class_set = combined_sample['Class']
    print(new_data)
    print(class_set)

    new_data_train, new_data_test, class_set_train, class_set_test = train_test_split(new_data,class_set, test_size=0.2, stratify=class_set, random_state=2)
   
    model = LogisticRegression()

    #training the model with the new_data_train and class_set_train
    model.fit(new_data_train, class_set_train)
    
    #accuracy score for training data and comparing it to the known frauds and determining how close this model is
    new_data_train_predicition = model.predict(new_data_train)
    trained_data_accuracy = accuracy_score(new_data_train_predicition, class_set_train)
    print('The accuracy score for the trained data is', trained_data_accuracy*100, '%')


    #accuracy test for testing data (0.8 of the data set)
    new_data_test_prediction = model.predict(new_data_test)
    test_data_accuracy_prediction = accuracy_score(new_data_test_prediction, class_set_test)
    print('The accuracy score for the tested data is ', test_data_accuracy_prediction*100 , '%')






logistic_regression()
