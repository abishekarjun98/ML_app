from flask import Flask,render_template,request, url_for, redirect
import numpy as np
import pickle

app = Flask(__name__)

saved_model=pickle.load(open('sample.pkl','rb'))

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/prediction',methods=['POST','GET'])
def prediction():

	sepal_length=request.form["sepal_length"]
	sepal_width=request.form["sepal_width"]
	petal_length=request.form["petal_length"]
	petal_width=request.form["petal_width"]

	given_input =np.array([[sepal_length,sepal_width,petal_length,petal_width]])

	prediction=saved_model.predict(given_input)

	return render_template("Home2.html", result=prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()