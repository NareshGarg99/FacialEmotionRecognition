import pandas as pd
import numpy as np
import flask
from flask import Flask,render_template, flash, request, url_for,redirect, session
import re
import os
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from skimage import io
from forms import RegistrationForm, LoginForm
from flask_uploads import UploadSet, configure_uploads, IMAGES



app = Flask(__name__)

photos = UploadSet('photos', IMAGES)

app.config['SECRET_KEY']='a7266cdf29030f092c955d59f1889390'
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'

configure_uploads(app,photos)


#load pretrained keras models
model =  load_model('facial_sentiment_trained_32000.h5')


@app.route('/home', methods=['GET', 'POST'])
@app.route('/',methods=['GET', 'POST'])
def home():
    return render_template("home.html")

@app.route('/contact',methods=['GET', 'POST'])
def contact():
	return render_template('contact.html', title='contact')

@app.route('/about',methods=['GET', 'POST'])
def about():
	return render_template('about.html', title='about')


@app.route("/register",methods=['GET', 'POST'])
def register():
	form = RegistrationForm()
	if form.validate_on_submit():
		flash('Account created for {form.username.data}!')
	return render_template("register.html", title='Register', form = form)

@app.route("/login", methods =["GET", "POST"])
def login():
	form = LoginForm()
	if form.validate_on_submit():
		if form.email.data == 'admin@blog.com' and form.password.data == 'password':
			flash('you have been logged in!', 'success')
			return redirect(url_for('home'))
		else:
			flash('login Unsuccessful. check username and password', 'danger')

	return render_template("login.html", title='Login', form= form)

@app.route('/upload', methods= ['POST', 'GET'])
def upload():
	if request.method=="POST":
		file_name = photos.save(request.files['photo'])
		img = io.imread(file_name, as_gray=True)
		array = tf.keras.preprocessing.image.img_to_array(img)
		image = tf.keras.preprocessing.image.array_to_img(array)
		im1 = image.resize((48,48))
		final_array = tf.keras.preprocessing.image.img_to_array(im1)
		final_final_array = (np.array(final_array)).reshape((1,) + final_array.shape)
		answer = model.predict_classes(final_final_array)
		mapp = {
			0 : 'Angry', 1:'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5 :'Suprise', 6: 'Neutral' 
		}
		final_answer = mapp[answer[0]]
	return render_template('home.html',sentiment=final_answer)
if __name__ == "__main__":
	app.run()
