# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 16:22:48 2020

@author: Kaushik Kumar
"""
#Import Dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
seed = 0 
import pickle

loan_train = pd.read_csv("C:\\Users\\HP\\Data Science\\Resume Projects\\Machine-learning\\Bank Interest Rate Prediction\\Notebook\\Preprocessed_Loan_Train.csv")

selected_features = ['Amount.Requested', 'Amount.Funded.By.Investors', 'Loan.Length', 'Loan.Purpose', 'Home.Ownership',
                     'State','Open.CREDIT.Lines', 'Inquiries.in.the.Last.6.Months', 'Fico']

X = loan_train[selected_features]
y = loan_train["Interest.Rate"]


#Encoding
    cat_to_num = ["Loan.Length","Loan.Purpose","State","Home.Ownership"]

for feat in cat_to_num:
     lbe = LabelEncoder()
     X[feat] = lbe.fit_transform(X[feat].values)
     pickle.dump(lbe, open('C:\\Users\\HP\\Data Science\\Resume Projects\\Machine-learning\\Bank Interest Rate Prediction\\Notebook\\encoding'+feat+'.pkl', 'wb'))

     print(lbe.classes_)


#Scaling
ss = StandardScaler()
X_columns = X.columns
X = pd.DataFrame(ss.fit_transform(X),columns = X_columns)
pickle.dump(ss, open('C:\\Users\\HP\\Data Science\\Resume Projects\\Machine-learning\\Bank Interest Rate Prediction\\Notebook\\scaler.pkl', 'wb'))

#Model Building
gbr = GradientBoostingRegressor(n_estimators=400,min_samples_split=9,min_samples_leaf= 24, 
                                max_depth= 13, learning_rate= 0.01,random_state=seed)
gbr.fit(X,y)

pickle.dump(gbr, open('C:\\Users\\HP\\Data Science\\Resume Projects\\Machine-learning\\Bank Interest Rate Prediction\\Notebook\\model.pkl', 'wb'))














