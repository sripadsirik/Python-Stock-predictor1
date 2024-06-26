import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors for hover functionality
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from ta import add_all_ta_features
from keras.regularizers import l1, l2
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import Adam
from scipy import stats


# Step 1: Download the data
ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2024-06-14'
data = yf.download(ticker, start=start_date, end=end_date)

# Check if data download was successful
if data.empty:
    print("Data download failed.")
    exit(1)

# Remove outliers
z_scores = np.abs(stats.zscore(data['Close']))
outliers = np.where(z_scores > 3)
outliers_to_drop = [index for index in outliers[0] if index in data.index]
data = data.drop(outliers_to_drop, axis=0)

# Step 2: Preprocess the data and add more features
data.dropna(inplace=True)
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")
data['Target'] = data['Close'].shift(-1)
data.fillna(0, inplace=True)

# Check if all rows were dropped during preprocessing
if data.empty:
    print("All rows were dropped during preprocessing.")
    exit(1)

# Step 3: Split the data
# Include more features
features = data.columns.drop('Target')
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0,1))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape X_train and X_test to be 3D, as expected by LSTM
X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Add the KerasRegressorWrapper class definition
class KerasRegressorWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, model):
        self.model = model
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions.flatten()

# Existing code
param_grid = {'n_estimators': [50, 100, 150], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [1, 2, 3]}
gbr = GradientBoostingRegressor(random_state=42, loss='squared_error')  # Change 'ls' to 'squared_error'
grid_search = GridSearchCV(gbr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
model1 = grid_search.best_estimator_

# Build the LSTM model with regularization
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

# Compile the LSTM model
model.compile(optimizer = Adam(learning_rate=0.01), loss = 'mean_squared_error')



# Fit the LSTM model
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Wrap the fitted LSTM model
wrapped_model = KerasRegressorWrapper(model)
wrapped_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train)

# Use an ensemble of models
# Include the LSTM model in the VotingRegressor
ensemble_model = VotingRegressor([('gbr', model1), ('lstm', wrapped_model)])
ensemble_model.fit(X_train, y_train)

# Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 7: Predict the stock price for Tomorrow
try:
    # Create a DataFrame with data for Tomorrow
    future_date = '2024-07-17'
    future_data = pd.DataFrame({feature: [data.iloc[-1][feature]] for feature in features})

    # Normalize the future data
    future_data = scaler.transform(future_data)

    # Use the trained model to predict the stock price for Tomorrow
    prediction = model.predict(future_data)
    print(f'Predicted Stock Price for Tomorrow: ${prediction[0][0]:.2f}')

except KeyError as e:
    print(f'Error accessing data for Tomorrow: {e}')

# Step 8: Plot the results with axis labels and hover functionality
plt.figure(figsize=(10, 6))
plt.plot(data.index[-len(y_test):], y_test.values, label='Actual Prices')  # Adjust indices for alignment
plt.plot(data.index[-len(predictions):], predictions, label='Predicted Prices')  # Adjust indices for alignment
plt.xlabel('Date')  # Label for the x-axis
plt.ylabel('Stock Price')  # Label for the y-axis
plt.title('Actual vs. Predicted Stock Prices')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend()

# Add hover functionality to display date and stock prices
def on_hover(sel):
    if isinstance(sel.target[0], np.float64):
        date = str(sel.target[0])  # Convert the float to string or use appropriate formatting
    else:
        date = sel.target[0].strftime('%Y-%m-%d')

mplcursors.cursor(hover=True).connect("add", on_hover)

plt.show()