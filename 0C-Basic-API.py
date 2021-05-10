from flask import Flask, request, jsonify
import numpy as np  
from tensorflow.keras.models import load_model
import joblib


#### THIS IS WHAT WE DO IN POSTMAN ###
# STEP 1: Create New Request
# STEP 2: Select POST
# STEP 3: Type correct URL (http://127.0.0.1:5000/api/wine)
# STEP 4: Select Body
# STEP 5: Select JSON
# STEP 6: Type or Paste in example json request
# STEP 7: Run 02-Basic-API.py to launch server and confirm the site is running
# Step 8: Run API request

def return_prediction(model,scaler,sample_json):
    
    # For larger data features, you should probably write a for loop
    # That builds out this array for you
    
    f_a = sample_json['fixed_acidity']
    v_a = sample_json['volatile_acidity']
    c_a = sample_json['citric_acid']
    r_s = sample_json['residual_sugar']
    chl = sample_json['chlorides']
    f_s_d = sample_json['free_sulfur_dioxide']
    t_s_d = sample_json['total_sulfur_dioxide']
    den = sample_json['density']
    ph = sample_json['pH']
    sul = sample_json['sulphates']
    alc = sample_json['alcohol']
    qua = sample_json['quality']
    
    wine = [[f_a,v_a,c_a,r_s,chl,f_s_d,t_s_d,den,ph,sul,alc,qua]]
    
    wine = scaler.transform(wine)
    
    classes = np.array(['red', 'white'])
    
    class_ind = model.predict_classes(wine)
    
    return classes[class_ind][0]


app = Flask(__name__)

@app.route('/')
def index():
    return '<h1>FLASK APP IS RUNNING!</h1>'


# REMEMBER TO LOAD THE MODEL AND THE SCALER!
wine_model = load_model("final_wine_model.h5")
wine_scaler = joblib.load("wine_scaler.pkl")

@app.route('/api/wine', methods=['POST'])
def predict_wine():

    content = request.json
    
    results = return_prediction(model=wine_model,scaler=wine_scaler,sample_json=content)
    
    return jsonify(results)

if __name__ == '__main__':
    app.run()