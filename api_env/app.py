import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
import os
import h5py

app = Flask(__name__)
with open('model_pkl','rb') as f:
    mp = pickle.load(f)

# path = os.path.dirname(__file__)
# print(path)
# MODEL = tf.keras.models.load_model(path + "\\test")



# train = pd.read_csv('train.csv')

# train.drop(['User_ID', 'Product_ID', 'Gender', 'City_Category', 
#             'Marital_Status', 'Product_Category_3'], axis = 1, inplace = True)
    
# train['Product_Category_2'].fillna(train['Product_Category_2'].median(), inplace = True)

# train['Product_Category_2'] = train['Product_Category_2'].astype('int')

# train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].apply(lambda x : str(x).replace('4+', '4'))

# train['Stay_In_Current_City_Years'] = train['Stay_In_Current_City_Years'].astype('int')

# train['Age'] = train['Age'].map(
#                             {'0-17' : 1,
#                              '18-25' : 2,
#                              '26-35' : 3,
#                              '36-45' : 4,
#                              '46-50' : 5,
#                              '51-55' : 6,
#                              '55+' : 7
#                              })

# X = train.drop('Purchase', axis = 1)
# Y = train['Purchase']

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# X = scaler.fit_transform(X)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ping')
def ping():
    return "Server is alive"

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        Category = request.form['Category']
        Discount = request.form['Discount']
        Profit = request.form['Profit']
        # Region = request.form['Region']
        # Category = request.form['Category']
    
    features = [Category,Discount,Profit]
        
    # int_features = [int(x) for x in features]
    final_features = [np.array(features)]
    prediction = mp.predict(final_features)

    return render_template('index.html', prediction_text='Sales: {}$'.format(round(prediction[0]) ))

if __name__ == "__main__":
    app.run(debug=True)






