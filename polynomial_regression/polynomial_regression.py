import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = [['Business Analyst', 1, 45000],
        ['Junior Consultant', 2, 50000],
        ['Senior Consultant', 3, 60000],
        ['Manager', 4, 80000],
        ['Country Manager', 5, 110000],
        ['Region Manager', 6, 150000],
        ['Partner', 7, 200000],
        ['Senior Partner', 8, 300000],
        ['C-level', 9, 500000],
        ['CEO', 10, 1000000]]
dataset = pd.DataFrame(data, columns=['Position', 'Level', 'Salary'])

dataset.head()


dataset.isna().sum()

x=dataset.iloc[:,1:-1].values
y=dataset.iloc[:,-1].values


sns.pairplot(dataset)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_ploy=poly_reg.fit_transform(x)
lin_reg_2=LinearRegression()
lin_reg_2.fit(x_ploy,y)



