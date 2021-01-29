from __future__ import print_function
from flask import Flask,render_template,request, url_for, redirect,flash
from werkzeug.utils import secure_filename
import numpy as np
import pandas as pd
import pickle
import os

import sys


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from sklearn import metrics


UPLOAD_FOLDER = 'D:\DB\Proj\data'
ALLOWED_EXTENSIONS = {'csv'}


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




sepal_length_new=[]
sepal_width_new=[]
petal_length_new=[]
petal_width_new=[]
species_new=[]


#model_list=['sample']
#pickle.dump(model_list,open("model_list.dat","wb"))
model_list=pickle.load(open('model_list.dat','rb'))

df=pd.read_csv("Train.csv")

def setup(DataFrame):
	DataFrame["species"]=DataFrame["species"].astype('category')
	DataFrame["species"]=DataFrame['species'].cat.codes
	x_original=DataFrame.iloc[:,0:4]
	y_original=DataFrame.iloc[:,-1:]
	return [x_original,y_original]
	
@app.route('/')
def home():
	return render_template("index.html",models=model_list)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            data_path=os.path.join(app.config['UPLOAD_FOLDER'], filename)
            append_data(data_path)
            return render_template("index.html",models=model_list)

def append_data(path):
	df_to_be_appended=pd.read_csv(path)
	global df
	df=df.append(df_to_be_appended,ignore_index=True)
	


@app.route('/simple_predict',methods=['POST','GET'])
def simple_predict():

	sepal_length=request.form["sepal_length"]
	sepal_width=request.form["sepal_width"]
	petal_length=request.form["petal_length"]
	petal_width=request.form["petal_width"]
	model_name=request.form["model"]

	model=pickle.load(open(model_name+'.pkl','rb'))

	given_input =np.array([[sepal_length,sepal_width,petal_length,petal_width]])

	prediction=model.predict(given_input)

	return render_template("result.html", result=prediction)



@app.route('/add_data',methods=['POST','GET'])
def add_data():

	sepal_length_new.append(request.form["sepal_length_new"])
	sepal_width_new.append(request.form["sepal_width_new"])
	petal_length_new.append(request.form["petal_length_new"])
	petal_width_new.append(request.form["petal_width_new"])
	species_new.append(request.form["species"])
	print(str(len(sepal_width_new))+"list", file=sys.stderr)
	return redirect(url_for('home'))




@app.route('/saving_model',methods=['POST','GET'])
def saving_model():

	data_columns=setup(df)

	x_train=data_columns[0]
	y_train=data_columns[1]
	
	custom_model_name=request.form["model_name"]
	model_list.append(custom_model_name)
	pickle.dump(model_list,open("model_list.dat","wb"))

	algo_name=request.form["algo_name"]

	
	x_new = pd.DataFrame({"sepal_length":sepal_length_new, "sepal_width":sepal_width_new,"petal_length":petal_length_new,"petal_width":petal_width_new}) 
	x_updated=x_train.append(x_new,ignore_index=True)
	

	print(str((x_updated).shape), file=sys.stderr)

	y_new=pd.DataFrame({"species":species_new})
	y_new=y_new.astype('int')
	y_updated=y_train.append(y_new,ignore_index=True)

	if algo_name=="Log_reg":

		user_model = LogisticRegression()

		user_model.fit(x_updated,y_updated)

	elif algo_name=="SVM":

		user_model =SVC(kernel="linear").fit(x_updated,y_updated)

	pickle.dump(user_model,open(custom_model_name+".pkl",'wb'))

	sepal_length_new.clear()
	sepal_width_new.clear()
	petal_length_new.clear()
	petal_width_new.clear()
	species_new.clear()

	return redirect(url_for('home'))





@app.route('/perfomance',methods=['POST','GET'])
def performance():

	df_test=pd.read_csv("Test.csv")
	data_test=setup(df_test)
	x_test=data_test[0]
	y_test=data_test[1]
	metrics_list={}
	for curr_model in model_list:
		model_=pickle.load(open(curr_model+'.pkl','rb'))
		y_pred = pd.Series(model_.predict(x_test))
		metrics_list[curr_model]=round(metrics.accuracy_score(y_test, y_pred),4)

	return render_template("metrics.html",metrics_list = metrics_list)




if __name__ == '__main__':
    app.debug = True
    app.run()	





