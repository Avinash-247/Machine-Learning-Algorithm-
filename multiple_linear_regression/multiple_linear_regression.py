import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import kagglehub

path_50_startups = kagglehub.dataset_download("farhanmd29/50-startups")

print("Path to 50 Startups dataset files:", path_50_startups)

dataset = pd.read_csv(f"{path_50_startups}/50_Startups.csv")

display(dataset.head())

dataset.isna().sum()

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encode', OneHotEncoder(),[3])], remainder="passthrough")
x=ct.fit_transform(x)

x

x=x[:,1:]

x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)



x_train


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)



y_pred=regression.predict(x_test)


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)

score
