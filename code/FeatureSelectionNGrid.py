import pandas as pd
import numpy as np 
from datetime import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler    
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

data = pd.read_csv('data/processed/SeoulDataTransf.csv')
df = data.copy()
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df.columns

df.drop(['Date', 'WeekDayEncoding'], axis=1, inplace=True)

# Encoding our categorical features. We will use Labeling Encoder for the binaries and One Hot Encoding for the multi labels
encoder = LabelEncoder()
df = pd.get_dummies(df, columns=['WeekDay', 'Seasons'])

x = df.copy()
y = x.pop('Rented Bike Count')
#y = y.values.ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# **Initial Feature Selection with L1 Regularization:**

# Standardize training features
scaler_initial = StandardScaler()
X_train_scaled = scaler_initial.fit_transform(X_train)
X_test_scaled = scaler_initial.transform(X_test)

# Train XGBRegressor with high L1 regularization
model_initial = XGBRegressor()
model_initial.fit(X_train_scaled, y_train)

# Analyze feature coefficients
feature_importances = model_initial.feature_importances_

# Identify features with low importance
low_importance_features = [
    feature
    for feature, importance in zip(x.columns, feature_importances)
    if importance < 0.01
]


# Evaluate the final model on the validation set
y_pred_initial = model_initial.predict(X_test_scaled)
rmse_initial = mean_squared_error(y_test, y_pred_initial, squared=False)
print("Initial Model Root Mean Squared Error:", rmse_initial)

mse = mean_squared_error(y_test, y_pred_initial)
r2 = r2_score(y_test, y_pred_initial)
print(f"Initial Mean Squared Error: {mse:.4f}")
print(f"Initial R-squared: {r2:.4f}")

sns.scatterplot(x=y_test, y=y_pred_initial)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.show()





# Reduce feature set
X_reduced = x.drop(low_importance_features, axis=1)

# **Build Final Model with Reduced Feature Set & Tuned Regularization:**
# Split the reduced data further for final model training and evaluation
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# Standardize training and validation features (create a new scaler)
scaler_final = StandardScaler()
X_train_final_scaled = scaler_final.fit_transform(X_train_final)
X_val_final_scaled = scaler_final.transform(X_val_final)

gsc = GridSearchCV(
            estimator=XGBRegressor(),
            param_grid = {
    "learning_rate": (0.15, 0.2),
    "max_depth": [7, 10, 15], 
    "min_child_weight": [5, 7], 
    "gamma": [0.1, 0.2],  
    "colsample_bytree": [0.8, 0.9], 
    "n_estimators": [250, 300], 
    "subsample": [0.9, 1],  
    "reg_alpha": [0, 0.1, 0.15],  
    "reg_lambda": [1e-3, 0.1], },
            cv=5, scoring='neg_mean_squared_error', verbose=2)

gsc.fit(X_train_final_scaled, y_train_final)

best_model = gsc.best_estimator_
best_params = gsc.best_params_

# Train the final XGBRegressor model with tuned parameters
model_final = best_model
model_final.fit(X_train_final_scaled, y_train_final)

# Evaluate the final model on the validation set
y_pred_final = model_final.predict(X_val_final_scaled)
rmse_final = mean_squared_error(y_val_final, y_pred_final, squared=False)
print("Final Model Root Mean Squared Error:", rmse_final)

mse = mean_squared_error(y_val_final, y_pred_final)
r2 = r2_score(y_val_final, y_pred_final)
print(f"Final Mean Squared Error: {mse:.4f}")
print(f"Final R-squared: {r2:.4f}")


sns.scatterplot(x=y_val_final, y=y_pred_final)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs. Predicted Values")
plt.grid(True)
plt.show()



# **Optional:** Invert scaling on predictions (if needed)
# y_pred_original_scale = scaler_final.inverse_transform(y_pred)




















'''param_grid={"learning_rate": (0.10, 0.15, 0.2),
                        "max_depth": [5, 7, 9],
                        "min_child_weight": [ 1, 3, 5, 7],
                        "gamma":[ 0.0, 0.1, 0.2],
                        "colsample_bytree":[0.5, 0.6],},'''