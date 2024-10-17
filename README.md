# EX 4 Implementation of Logistic Regression Model to Predict the Placement Status of Student
## DATE:

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.  Data Collection and Preprocessing
2.  Algorithm for Logistic Regression Prediction
3.  Explanation of Key Components
4.  Evaluate the model

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Livya Dharshini G
RegisterNumber: 2305001013
*/
import pandas as pd
data=pd.read_csv("/content/ex45Placement_Data.csv")
data.head()
data1=data.copy()
data1.head()
data1=data1.drop(['sl_no','salary'],axis=1)
data1
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
y=data1.iloc[:,-1]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_pred,x_test
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("accuracy score:",accuracy)
print("\nconfusion matrix:\n",confusion)
print("\nclassification report:\n",cr)
from sklearn import metrics
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=confusion)
cm_display.plot()
```

## Output:![image](https://github.com/user-attachments/assets/56ac87f9-ab58-4207-a73c-31104e196582)
![image](https://github.com/user-attachments/assets/91849dc0-a7e3-44b8-8193-47e3fcc302bf)
![image](https://github.com/user-attachments/assets/83a56325-b0aa-46e8-8500-9359a7a1e872)
![image](https://github.com/user-attachments/assets/85f42a35-15f6-4724-81cc-d08a5ecf1c16)
![image](https://github.com/user-attachments/assets/c63f5621-4c95-4012-abb0-83c704f04229)
![image](https://github.com/user-attachments/assets/1b3cc96c-5a9d-4d4a-89c2-e739cd6ce7e3)
![image](https://github.com/user-attachments/assets/2af27ce0-f110-4308-b9d4-40a1074f9fd0)
![image](https://github.com/user-attachments/assets/bd18b400-376c-4290-bd0e-a5020b6b687f)
![image](https://github.com/user-attachments/assets/dd3deecb-6044-4573-945a-7ebc373f1ea0)
![image](https://github.com/user-attachments/assets/c534973d-25e7-4c51-a3f2-500eac1d32ad)
![image](https://github.com/user-attachments/assets/3f5b3a18-3e2e-4ae1-8152-ef0cf1a70004)
![image](https://github.com/user-attachments/assets/25547a39-ff33-4989-ae1c-5634fa4f84c5)














## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
