from flask import Flask,render_template,session,url_for,redirect
import numpy as np 
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField
from tensorflow.keras.models import load_model
import joblib 

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
app.config['SECRET_KEY'] = 'mysecretkey'

class WineForm(FlaskForm):

    f_a = TextField('fixed_acidity')
    v_a = TextField('volatile_acidity')
    c_a = TextField('citric_acid')
    r_s = TextField('residual_sugar')
    chl = TextField('chlorides')
    f_s_d = TextField('free_sulfur_dioxide')
    t_s_d = TextField('total_sulfur_dioxide')
    den = TextField('density')
    ph = TextField('pH')
    sul = TextField('sulphates')
    alc = TextField('alcohol')
    qua = TextField('quality')

    submit = SubmitField('Analyze')


@app.route("/",methods=['GET','POST'])
def index():

	form = WineForm()

	if form.validate_on_submit():

		session['f_a'] = form.f_a.data
		session['v_a'] = form.v_a.data
		session['c_a'] = form.c_a.data
		session['r_s'] = form.r_s.data
		session['chl'] = form.chl.data
		session['f_s_d'] = form.f_s_d.data
		session['t_s_d'] = form.t_s_d.data
		session['den'] = form.den.data
		session['ph'] = form.ph.data
		session['sul'] = form.sul.data
		session['alc'] = form.alc.data
		session['qua'] = form.qua.data 

		return redirect(url_for("prediction"))

	return render_template('home.html',form=form)
	


wine_model = load_model("final_wine_model.h5")
wine_scaler = joblib.load("wine_scaler.pkl")

@app.route('/prediction')
def prediction():
	
	content = {}

	content['fixed_acidity'] = float(session['f_a'])
	content['volatile_acidity'] = float(session['v_a'])
	content['citric_acid'] = float(session['c_a'])
	content['residual_sugar'] = float(session['r_s'])
	content['chlorides'] = float(session['chl'])
	content['free_sulfur_dioxide'] = float(session['f_s_d'])
	content['total_sulfur_dioxide'] = float(session['t_s_d'])
	content['density'] = float(session['den'])
	content['pH'] = float(session['ph'])
	content['sulphates'] = float(session['sul'])
	content['alcohol'] = float(session['alc'])
	content['quality'] = float(session['qua'])

	results = return_prediction(werwine_model,wine_scaler,content)

	return render_template('prediction.html',results=results)


if __name__=='__main__':
	app.run()