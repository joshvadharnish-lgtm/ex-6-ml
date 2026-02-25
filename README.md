# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset, separate features and target, scale the input features, and encode the target labels.

2.Split the dataset into training and testing sets using stratified sampling.

3.Train the Logistic Regression model using training data and predict the test data.

4.Evaluate the model using accuracy, confusion matrix, and classification report, then visualize the confusion matrix.


## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: G.DHARNISH
RegisterNumber:  25004380
*/


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset 
df=pd.read_csv("food_items.csv")
#inspect the dataset
print("Dataset Overview")
print(df.head())
print("\ndatset Info")
print(df.info())

X_raw=df.iloc[:, :-1]
y_raw=df.iloc[:, -1:]
X_raw

scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

penalty='l2'
multi_class='multnomial'
solver='lbfgs'
max_iter=1000

model = LogisticRegression(max_iter=2000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:
<img width="642" height="540" alt="Screenshot 2026-02-25 094846" src="https://github.com/user-attachments/assets/4517ef48-a833-4777-aadd-863dccf48279" />


<img width="624" height="358" alt="Screenshot 2026-02-25 094837" src="https://github.com/user-attachments/assets/99cd7860-ba6c-4df1-a720-83282400e846" />

<img width="828" height="663" alt="Screenshot 2026-02-25 094814" src="https://github.com/user-attachme![Uploading Screenshot 2026-02-25 094800.png…]()
nts/assets/6c1930b6-2ab2-494c-8abc-37777f0949d9" />
<img width="640" height="660" alt="Screenshot 2026-02-25 094830" src="https://github.com/user-attachments/assets/ac34203d-08ed-4d2d-a059-2f017866d7ac" />

## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
