import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
import joblib

#load data
X, y = load_diabetes(return_X_y=True)
df = pd.DataFrame(X, columns=['age', 'sex', 'bmi', 'bp', 's1', 's2','s3','s4','s5','s6'])
df['target'] = y

#spilt data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#train with linear regression
lr = LinearRegression().fit(X_train, y_train)

#evaluate the model
preds = lr.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print("RMSE: ", rmse)

#save the model for fastapi
joblib.dump(lr, "model/diabetes_model.pkl")

