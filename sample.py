import numpy as np
import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

df=pd.read_csv("iris.csv")

df["species"]=df["species"].astype('category')
df["species"]=df['species'].cat.codes

x=df.iloc[:,0:4]
y=df.iloc[:,-1:]

X_train,X_test,Y_train,Y_test= train_test_split(x,y,test_size=0.4)

model = LogisticRegression()

model.fit(X_train,Y_train)

#print(model.predict(X_test))

pickle.dump(model,open('sample.pkl','wb'))