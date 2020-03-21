import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    month_map = {"Jan":2, "May": 6, "Nov": 7, "Mar": 5, "Dec": 1, "Oct": 8, "Sep": 9, "Aug": 0, "Jul": 3, "June": 4, "Feb": 2}

    data = np.array([x for x in request.form.values()])
    column = ["PageValues", "ExitRates", "Administrative", "Month", "ProductRelated_Duration"]
    x_test_request = pd.DataFrame(data.reshape(1, 5), columns=column)

    #Label Encoding
    x_test_request["Month"] = month_map[x_test_request.iloc[0]["Month"]]

    # #Scaling
    # for i in column:
    #     x_test_request[i] = (float(x_test_request.iloc[0][i])-float(scaling[i][0]))/float(scaling[i][1])

    prediction = model.predict(x_test_request)

    if prediction:
        return render_template('index.html', prediction_text='This Customer will generate revenue to the company')
    else:
        return render_template('index.html', prediction_text='This Customer will not generate revenue to the company')


if __name__ == "__main__":
    app.run(debug=True)