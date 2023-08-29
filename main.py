#detecting cc fraud through an unbalanced data set(which is
#not great to train a model because it is one class to heavy)
#the use of linear regression, and then some other type, and then some other type

import numpy as np #allows mathmatical functions for multi-dimensional arrays
import pandas as pd #allows for data frames 
from sklearn.model_selection import train_test_split #allows for training and testing data
from sklearn.linear_model import LinearRegression #allows for predicition classification of either the pos or neg class
from sklearn.metrics import accuracy_score #allows determination of how accurate the model is

def linear_regression():
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

linear_regression()

def train_data():
    print('Hello World')

train_data()