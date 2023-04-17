from flask import Flask, jsonify, request, render_template
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)


# Load the machine learning model
model = tf.keras.models.load_model('model.h5')


@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle incoming requests


@app.route('/predict', methods=['POST'])
def predict():
    Profit = request.form['Profit']
    Quantity = (request.form['Quantity'])
    Category_Furniture = (request.form['Category_Furniture'])
    Category_Office_Supplies = (request.form['Category_Office_Supplies'])
    Category_Technology = (request.form['Category_Technology'])
    
    # features = [Profit, Quantity, Category_Furniture,
    #             Category_Office_Supplies, Category_Technology]
    # # int_features = [int(x) for x in features]
    # # final_features = [np.array(int_features)]
    # prediction = model.predict(features)
    # Profit = request.args.get('Profit')
    # Quantity = request.args.get('Quantity')
    # Category_Furniture = request.args.get('Category_Furniture')
    # Category_Office_Supplies = request.args.get('Category_Office_Supplies')
    # Category_Technology = request.args.get('Category_Technology')
    
    features = [Profit, Quantity, Category_Furniture, Category_Office_Supplies, Category_Technology]
    int_features = [float(x) for x in features]
    prediction = model.predict(np.array(int_features))

    return render_template('index.html', prediction_text='Sales: {}$'.format(round(prediction[0])))


if __name__ == "__main__":
    app.run(debug=True)
