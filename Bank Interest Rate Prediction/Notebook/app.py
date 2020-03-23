# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 14:16:41 2020

@author: Kaushik Kumar
"""

import numpy as np
from flask import Flask, request, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    data = np.array([x for x in request.form.values()])
    #data = np.array([25000,25000,'60 months','debt_consolidation','MORTGAGE','VA',11,3,720])
    
    selected_features = ['Amount.Requested', 'Amount.Funded.By.Investors', 'Loan.Length', 'Loan.Purpose', 'Home.Ownership',
                     'State','Open.CREDIT.Lines', 'Inquiries.in.the.Last.6.Months', 'Fico']
    x_test_request = pd.DataFrame(data.reshape(1, len(selected_features)), columns=selected_features)

    #Label Encoding
    cat_to_num = ["Loan.Length","Loan.Purpose","State","Home.Ownership"]
    for feat in cat_to_num:
        encoder = pickle.load(open('C:\\Users\\HP\\Data Science\\Resume Projects\\Machine-learning\\Bank Interest Rate Prediction\\Notebook\\Encoders\\encoding'+feat+'.pkl','rb'))
        
        x_test_request[feat] = encoder.transform(x_test_request[feat])
    
    #Scaling
    x_test_request = scaler.transform(x_test_request)
    

    prediction = model.predict(x_test_request)

    return render_template('index.html', prediction_text='This Customer is liable to an interest rate of '+ str(round(prediction[0],2)) + '%')
    


if __name__ == "__main__":
    app.run(debug=True)

