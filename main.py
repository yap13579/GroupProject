import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os 


os.system('cls')
# Fetch historical stock data
company = '1155.KL'

# Define a start date and End Date
start = date.datetime(2018, 1, 1)
end = date.datetime(2023, 1, 1)

# Read Stock Price Data 
data = yf.download(company, start, end)

# Feature Engineering
data['Target'] = data['Close'].shift(-1)  # Predicting the next day's Close price

# Drop rows with NaN values
data.dropna(inplace=True)

# Features and target variable
X = data[['Open', 'High', 'Low', 'Close', 'Volume']]  # You can include more features if needed
y = data['Target']

# Normalizing the data
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
y_normalized = scaler.fit_transform(np.array(y).reshape(-1, 1))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.2, random_state=42)

# Create and train the MLP Regressor
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
predictions = model.predict(X_test)

# Reverse normalization for predictions
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_orig = scaler.inverse_transform(y_test)

# Evaluate the model
r2 = r2_score(y_test_orig, predictions)
mse = mean_squared_error(y_test_orig, predictions)

print(f"R-squared Score: {r2}")
print(f"Mean Squared Error: {mse}")

# Plotting actual vs predicted values (scatter plot)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_orig, predictions, color='blue', alpha=0.5)
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.show()

# Plotting actual vs predicted values (line graph)
plt.figure(figsize=(12, 6))
plt.plot(data.index[-len(y_test_orig):], y_test_orig, label='Actual Prices', color='green')
plt.plot(data.index[-len(predictions):], predictions, label='Predicted Prices', color='red')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()

start = date.datetime(2024, 1, 1)
end = date.datetime.today()
# Read Stock Price Data 
new_data = yf.download(company, start, end)

# Feature Engineering
new_data['Target'] = new_data['Close'].shift(-1)  # Predicting the next day's Close price

# Handle NaN values
new_data.fillna(method='ffill', inplace=True)
# Features and target variable
X_new = new_data[['Open', 'High', 'Low', 'Close', 'Volume']]  # You can include more features if needed
y_new = new_data[['Target']][:-1]

# Normalizing the new data
scaler = MinMaxScaler()
X_new_normalized = scaler.fit_transform(X_new)
y_new_normalized = scaler.fit_transform(np.array(y_new).reshape(-1, 1))

# Predict using the trained model
predictions_new = model.predict(X_new_normalized)

# Reverse normalization for predictions
predictions_new = scaler.inverse_transform(predictions_new.reshape(-1, 1))
y_new_orig = scaler.inverse_transform(y_new_normalized)

# Calculate R-squared Score and Mean Squared Error for the new predictions
r2_new = r2_score(y_new_orig, predictions_new[:-1])
mse_new = mean_squared_error(y_new_orig, predictions_new[:-1])

print(f"R-squared Score for new data: {r2_new}")
print(f"Mean Squared Error for new data: {mse_new}")
lastdate=new_data.index[-1]
newdate=lastdate+pd.DateOffset(days=1)
newrow=pd.DataFrame(index=[newdate],columns=new_data.columns)
new_data=pd.concat([new_data,newrow])
# Plotting predicted values for the new data
print(new_data)
print(predictions_new)
plt.figure(figsize=(12, 6))
plt.plot(new_data.iloc[:-1].index, new_data[['Target']][:-1], label='Actual Prices', color='green')
plt.plot(new_data.iloc[:-1].index, predictions_new, label='Predicted Prices', color='red')
#plt.plot(new_data.tail(1).index, predictions_new[-1:], label='Future Prices', color='blue')
plt.title('Actual vs Predicted Stock Prices for New Data')
plt.xlabel('Date')
plt.ylabel('Stock Prices')
plt.legend()
plt.show()