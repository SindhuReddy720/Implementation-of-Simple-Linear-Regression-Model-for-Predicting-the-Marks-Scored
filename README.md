# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: pelleti sindhu sri
RegisterNumber: 212224240113

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "Marks": [25, 45, 50, 60, 65, 70, 75, 85, 95]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)
X = df[['Hours']]
Y = df['Marks']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
print("\nPredicted values for test set:", Y_pred)
hours = np.array([[7.5]])
predicted_marks = model.predict(hours)
print(f"\nPredicted Marks for 7.5 study hours = {predicted_marks[0]:.2f}")
plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("Hours Studied")
plt.ylabel("Marks Scored")
plt.title("Simple Linear Regression - Predicting Student Marks")
plt.legend()
plt.show()

*/
```

## Output:
<img width="756" height="560" alt="image" src="https://github.com/user-attachments/assets/03f6393a-27ad-49a5-94bc-dbf6a9295ae1" />



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
