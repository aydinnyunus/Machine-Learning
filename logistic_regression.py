from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

df = pd.read_csv('W.csv',sep = ',')
df['Height']= df.Height*2.54
df['Weight'] = df.Weight*0.45
Gender = df[df.columns[0]]
HW = df[df.columns[1:3]]

'''
Gender = df['Gender']
Features = df['Height']

#[height, weight, shoe_size]
'''
'''
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43],[185,65,45]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male','male']
'''

log = LogisticRegression().fit(HW, Gender)
#lin = LinearRegression().fit(HW, Gender)
print("Accuracy score",log.score(HW, Gender))

print(log.predict([[150,75]]))
#print("Linear",lin.predict([[150,75]]))

print(log.predict([[194,95]]))
print(log.predict([[164,45]]))
print(log.predict([[144,55]]))
print(log.predict([[185,85]]))
print(log.predict([[175,65]]))
print(log.predict([[175,77]]))
print(log.predict([[175,97]]))
print(log.predict([[215,115]]))



