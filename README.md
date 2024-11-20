# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the `pandas` library and read the dataset `Placement_Data.csv` into a DataFrame `data`.
2. Display the first few rows of `data` using `data.head()` to check its contents.
3. Create a copy of `data` named `data1` to work on.
4. Drop the columns `"sl_no"` and `"salary"` from `data1` using `data1.drop()`.
5. Display the first few rows of `data1` after dropping columns to confirm changes.
6. Check for missing values in `data1` using `data1.isnull().sum()`.
7. Check for duplicate rows in `data1` using `data1.duplicated().sum()`.
8. Import `LabelEncoder` from `sklearn.preprocessing` for encoding categorical variables.
9. Create an instance of `LabelEncoder` named `le`.
10. Encode categorical columns (`"gender"`, `"ssc_b"`, `"hsc_b"`, `"hsc_s"`, `"degree_t"`, `"workex"`, `"specialisation"`, `"status"`) in `data1` using `le.fit_transform()`.
11. Display `data1` after encoding to check that categorical columns are encoded into numerical values.
12. Extract the independent variables `x` (all columns except `"status"`) from `data1`.
13. Extract the dependent variable `y` (the `"status"` column) from `data1`.
14. Split `x` and `y` into training and testing sets using `train_test_split()` with 80% training and 20% testing data, setting `random_state` to 0 for reproducibility.
15. Import `LogisticRegression` from `sklearn.linear_model`.
16. Create an instance of `LogisticRegression` named `lr` with the solver set to `"liblinear"`.
17. Train the model on the training data using `lr.fit()` with `x_train` and `y_train`.
18. Make predictions on the test set `x_test` using `lr.predict()` and store them in `y_pred`.
19. Print or return `y_pred` to display the predicted output.

## Program:
```py
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by:Dhivya Dharshini B
RegisterNumber: 2122232340031
import pandas as pd
data1=pd.read_csv('Placement_Data.csv')
data1.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
x=data1.iloc[:,:-1]
x
y=data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression(solver ="liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
*/
```
## Output:

### op1:
![image](https://github.com/user-attachments/assets/71453f77-4b5f-495f-9436-9cd0769f5d31)
### op2:
![image](https://github.com/user-attachments/assets/0ecc2a49-9112-480d-a1b5-969672e84aa1)
### op3:
![image](https://github.com/user-attachments/assets/85237170-99eb-4cb2-8bcd-326d65c99e77)
### op4:
![image](https://github.com/user-attachments/assets/15bbe57f-6248-4e6d-940d-4279ddde2e5e)
### op5:
![image](https://github.com/user-attachments/assets/3f796858-aea1-46bc-a189-9879d8cad314)
### op6:
![image](https://github.com/user-attachments/assets/7f4fb20e-21ea-4606-9d0b-bbc5ad79b708)
### op7:
![image](https://github.com/user-attachments/assets/efe334fa-2401-48a4-a392-1f8e42490bb1)
### op8:
  ![image](https://github.com/user-attachments/assets/7e77fc00-54b9-43c7-b259-e38e51e8ed68)

### op9:
![image](https://github.com/user-attachments/assets/0e2924aa-cd18-426e-af28-fb5e9efb7df5)

### op10:
![image](https://github.com/user-attachments/assets/ea84e14a-35d4-4739-a2bc-4e67e79a23a2)

### op11:
![image](https://github.com/user-attachments/assets/f2e82f7d-81cb-40c1-9ef8-161a5772d603)
### op12:
![image](https://github.com/user-attachments/assets/4d8184c1-0f65-416b-a357-a64de88aa7a8)





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
