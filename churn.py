import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
# Load dataset
data = pd.read_csv('Churn-Data.csv')

# Handle missing values if any
data.replace(' ', np.nan, inplace=True)  # Replace empty strings with NaN

data["gender"]=data["gender"].map({"Male":0,"Female":1})
data["Partner"]=data["Partner"].map({"Yes":1,"No":0})
data["Dependents"]=data["Dependents"].map({"Yes":1,"No":0})
data["PhoneService"]=data["PhoneService"].map({"Yes":1,"No":0})
data["MultipleLines"]=data["MultipleLines"].map({"Yes":2,"No":1,"No phone service":0})
data["InternetService"]=data["InternetService"].map({"DSL":2,"Fiber optic":1,"No":0})
data["OnlineSecurity"]=data["OnlineSecurity"].map({"Yes":2,"No":1,"No internet service":0})
data["OnlineBackup"]=data["OnlineBackup"].map({"Yes":2,"No":1,"No internet service":0})
data["DeviceProtection"]=data["DeviceProtection"].map({"Yes":2,"No":1,"No internet service":0})
data["TechSupport"]=data["TechSupport"].map({"Yes":2,"No":1,"No internet service":0})
data["TV_Streaming"]=data["TV_Streaming"].map({"Yes":2,"No":1,"No internet service":0})
data["Movie_Streaming"]=data["Movie_Streaming"].map({"Yes":2,"No":1,"No internet service":0})
data["Contract"]=data["Contract"].map({"Month-to-month":0,"One year":1,"Two year":2})
data["PaperlessBilling"]=data["PaperlessBilling"].map({"Yes":1,"No":0})
data["Method_Payment"]=data["Method_Payment"].map({"Electronic check":0,"Mailed check":1,"Bank transfer (automatic)":2,"Credit card (automatic)":3})

data["gender"]=data["gender"].fillna(np.mean(data["gender"],axis=0))
data["SeniorCitizen"]=data["SeniorCitizen"].fillna(np.mean(data["SeniorCitizen"],axis=0))
data["Partner"]=data["Partner"].fillna(np.mean(data["Partner"],axis=0))
data["Dependents"]=data["Dependents"].fillna(np.mean(data["Dependents"],axis=0))
data["tenure"]=data["tenure"].fillna(np.mean(data["tenure"],axis=0))
data["PhoneService"]=data["PhoneService"].fillna(np.mean(data["PhoneService"],axis=0))
data["MultipleLines"]=data["MultipleLines"].fillna(np.mean(data["MultipleLines"],axis=0))
data["InternetService"]=data["InternetService"].fillna(np.mean(data["InternetService"],axis=0))
data["OnlineSecurity"]=data["OnlineSecurity"].fillna(np.mean(data["OnlineSecurity"],axis=0))
data["OnlineBackup"]=data["OnlineBackup"].fillna(np.mean(data["OnlineBackup"],axis=0))
data["DeviceProtection"]=data["DeviceProtection"].fillna(np.mean(data["DeviceProtection"],axis=0))
data["TechSupport"]=data["TechSupport"].fillna(np.mean(data["TechSupport"],axis=0))
data["TV_Streaming"]=data["TV_Streaming"].fillna(np.mean(data["TV_Streaming"],axis=0))
data["Movie_Streaming"]=data["Movie_Streaming"].fillna(np.mean(data["Movie_Streaming"],axis=0))
data["Contract"]=data["Contract"].fillna(np.mean(data["Contract"],axis=0))
data["PaperlessBilling"]=data["PaperlessBilling"].fillna(np.mean(data["PaperlessBilling"],axis=0))
data["Method_Payment"]=data["Method_Payment"].fillna(np.mean(data["Method_Payment"],axis=0))
data["TotalCharges"]=data["TotalCharges"].fillna(np.mean(data["Method_Payment"],axis=0))


# Split dataset into features and target variable
X = data.drop(columns=['Churn', 'cID'])
y = data['Churn']

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Model training
model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,activation="logistic", random_state=42)
model.fit(X_train_scaled, y_train)

# Model evaluation
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, pos_label='Yes')
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Classification Report:")
print(classification_report(y_test, y_pred))
