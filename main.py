# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 10:22:39 2021

@author: suganya arvind
"""

from flask import Flask, render_template, request
import numpy as np
import pickle


cv_transformer = pickle.load(open('CountVectorizer.pkl','rb'))
model = pickle.load(open('NBmodel.pkl','rb'))

main = Flask(__name__)

@main.route('/')
def home():
    return render_template('index.html')


@main.route('/predict', methods=['POST'])
def predict():
    #print("I am predicted")
    if request.method =='POST':
        inp = request.form['message']
        value = cv_transformer.transform([inp]).toarray()
        output = model.predict(value)[0]
        return render_template('index.html', prediction=output)

if __name__=='__main__':
    main.run(debug=True)