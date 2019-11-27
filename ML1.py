import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#Read the data
df = pd.read_fwf("brain_body.txt")
#X axis is Brain
x = df[['Brain']]
#Y axis is Body
y = df[['Body']]

#Print first 5 row of data
print(df.head())

#Return to linear regression and fit x,y
regression = linear_model.LinearRegression()
regression.fit(x,y)

#Sketch the graph and predict x value.
plt.scatter(x,y)
plt.plot(x,regression.predict(x))

#Print the predict x
print(regression.predict(x))
print("----------------------")
#Print the predict y
print(regression.predict(y))
plt.show()

