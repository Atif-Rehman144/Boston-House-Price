import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)
## Load the model
rfmodel=pickle.load(open('RandomForest.pkl','rb'))
scalar=pickle.load(open('scaling.pkl','rb'))
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=rfmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    scaled_inp=scalar.transform(np.array(data).reshape(1,-1))
    print(scaled_inp)
    predicted=rfmodel.predict(scaled_inp)
    return render_template('home.html',prediction_text="The house price is(in 1000s dollars) {}".format(predicted))


if __name__=="__main__":
    app.run(debug=False)
    ## changed to FALSE as its good practice
   
     