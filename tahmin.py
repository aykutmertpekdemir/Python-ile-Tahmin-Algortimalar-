# -*- coding: utf-8 -*-

 # -*- coding: utf-8 -*-
"""
Aykut Mert Pekdemir

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


veriler = pd.read_excel("tahmin.xlsx")

x = veriler.iloc[:,1:2].values
y = veriler.iloc[:,2:].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

plt.scatter(x,y,color='blue')
plt.plot(x,lin_reg.predict(x), color = 'red')
plt.show()

print("Linear R2 degeri:")
print(r2_score(y, lin_reg.predict((x))))

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 6)
x_poly = poly_reg.fit_transform(x)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)
plt.scatter(x,y,color = 'red')
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.show()


print("Polynomial R2 degeri:")
print(r2_score(y, lin_reg2.predict(poly_reg.fit_transform(x)) ))

#verilerin olceklenmesi
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(x)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(y)

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )

from sklearn.tree import DecisionTreeRegressor 
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(x,y)
Z = x + 0.5
K = x - 0.4
plt.scatter(x,y, color='red')
plt.plot(x,r_dt.predict(x), color='blue')
plt.plot(x,r_dt.predict(Z),color='green')
plt.plot(x,r_dt.predict(K), color = 'yellow')
plt.show()
print(r_dt.predict([[6.6]]))

print("Decision Tree R2 degeri:")
print(r2_score(y, r_dt.predict(x)) )

from sklearn.ensemble import RandomForestRegressor  
rf_reg = RandomForestRegressor(n_estimators = 10, random_state=0)
rf_reg.fit(x,y)

print(rf_reg.predict([[6.6]]))

plt.scatter(x,y, color='red')
plt.plot(x,rf_reg.predict(x), color = 'blue')
plt.plot(x,rf_reg.predict(Z), color = 'green')
plt.plot(x,r_dt.predict(K), color = 'yellow')
plt.show()

print("Random Forest R2 degeri:")
print(r2_score(y, rf_reg.predict(x)) )
print(r2_score(y, rf_reg.predict(K)) )
print(r2_score(y, rf_reg.predict(Z)) )

#R2 degerleri
print('----------------')
print("Linear R2 degeri:")
print(r2_score(y, lin_reg.predict((x))))


print("Polynomial R2 degeri:")
print(r2_score(y, lin_reg2.predict(poly_reg.fit_transform(x)) ))


print("SVR R2 degeri:")
print(r2_score(y_olcekli, svr_reg.predict(x_olcekli)) )

print("Decision Tree R2 degeri:")
print(r2_score(y, r_dt.predict(x)) )

print("Random Forest R2 degeri:")
print(r2_score(y, rf_reg.predict(x)) )
