from flask import Flask,request,jsonify,render_template
import numpy as np
import pickle
model = pickle.load(open('models/CDRmodel_RF.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')
#render_template('index.html')

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

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
    print(int_features)
    int_features_i = int_features[0:2]
    int_features_ii = int_features[0]
    if int(int_features_ii) > 30000000 and int(int_features_ii) <= 39999999:
        int_features_ii = 1
    else:
        int_features_ii = 0    

    addition_features_i = np.array([0])
    
    int_features_iii = int_features[2]
    if int_features_iii == '':
        int_features_iii = 0
    if int_features_iii == "on":
        int_features_iii = 1
    
    addition_features_ii = np.array([0,1])
    
    int_features = np.concatenate((int_features_i, int_features_ii, addition_features_i, int_features_iii, addition_features_ii), axis=None)
    print(int_features)
    features = [np.array(int_features)]
    prediction = model.predict(features)
    
    output = prediction[0]
    numOut = int_features[0]
    
    if output == "Yes":
        output = "a normal Call."
    else:
        output = "a fraudulent Call."
    
    return render_template('result.html', prediction_text='The call is {}'.format(output), number=numOut)

#jsonify({'The call is':output})

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    print(data_unseen)
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=105)
