# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 20:42:59 2023

@author: Cheun
"""

from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
model = pickle.load(open('models/CDRmodel_RF.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
#render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    #phoneNo = request.form.get('callingNum')
    #callphoneNo = request.form.get('calledNum')
    #input_query = np.array([[phoneNo, callphoneNo]])
    #addition_features = np.array([0,0,0,0,1])
    #int_features = np.append(input_query, addition_features, axis=None)
    #result = model.predict(input_query)[0]
    #return jsonify({'Normal':str(result)})
    int_features = [x for x in request.form.values()]
    int_features = int_features[0:2]
    addition_features = np.array([0,0,0,0,1])
    int_features = np.append(int_features, addition_features, axis=None)
    print(int_features)
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    output = prediction[0]
    
    if output == "Yes":
        output = "a normal Call."
    else:
        output = "a fraudulent Call."
    
    return render_template('index.html', prediction_text='The call is {}'.format(output))

#jsonify({'The call is':output})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)