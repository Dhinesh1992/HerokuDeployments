from flask import Flask, request, render_template
from flask import Markup

import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)

transformer = pickle.load(open('column_transformer.pkl','rb'))
model = pickle.load(open('Rf_model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    Brand_name=request.form.get("Brand")
    Transmission=request.form.get("Transmission")
    Fuel_Type=request.form.get("Fuel_Type")
    Owner_Type=request.form.get("owner")
    Location=request.form.get("Location")
    Cars=request.form.get("cars")
    Year=request.form.get("Year")
    Kilometers_Driven=request.form.get("Kilometers_Driven")
    Mileage=request.form.get("Mileage")
    Engine=request.form.get("Engine")
    Power=request.form.get("Power")
    seat=request.form.get("Seats")
    
    df = pd.DataFrame([[Brand_name, Transmission, Fuel_Type, Owner_Type, Location, Cars, Year, Kilometers_Driven, Mileage, Engine, Power, seat]])
    input = transformer.transform(df)
    
    output = round(model.predict(input)[0],2)    
    
    value = Markup('Predicted price of the car is {} Lakhs'.format(output))
    
    #return render_template('home.html',prediction_text=[Brand_name, Transmission, Fuel_Type, Owner_Type, Location, Cars, Year, Kilometers_Driven, Mileage, Engine, Power, seat])

    return render_template('home.html',prediction_text=value)


if __name__=='__main__':
    app.run(debug=True)

