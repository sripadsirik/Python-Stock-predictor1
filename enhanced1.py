import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors for hover functionality
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import ta


# Step 1: Download the data
try:
    ticker = 'NVDA'
    start_date = '2020-01-01'
    end_date = '2024-06-16'
    data = yf.download(ticker, start=start_date, end=end_date)
except Exception as e:
    print(f'Error downloading data: {e}')
    exit(1)

# Check if data is empty
if data.empty:
    print('No data downloaded. Please check your internet connection or try again later.')
    exit(1)

# Step 2: Preprocess the data
data.dropna(inplace=True)
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Calculate RSI
data['RSI'] = ta.momentum.rsi(data['Close'])

# Calculate MACD
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd_diff()

data['Volume'] = data['Volume']
data['Open'] = data['Open']
data['High'] = data['High']
data['Low'] = data['Low']
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Step 3: Split the data
features = ['Close', 'MA20', 'MA50', 'RSI', 'MACD', 'Volume', 'Open', 'High', 'Low']
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build the model
model = GradientBoostingRegressor()
parameters = {
    'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'max_depth': [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'min_samples_split': [2, 5, 10, 15, 20],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10]
}
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Step 5: Evaluate the model
predictions = best_model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 6: Predict the stock price for Tomorrow
try:
    # Create a DataFrame with data for Tomorrow
    future_date = '2024-07-17'
    future_data = pd.DataFrame({'Close': [data.iloc[-1]['Close']],
                                'MA20': [data.iloc[-1]['MA20']],
                                'MA50': [data.iloc[-1]['MA50']],
                                'Volume': [data.iloc[-1]['Volume']],
                                'Open': [data.iloc[-1]['Open']],
                                'High': [data.iloc[-1]['High']],
                                'Low': [data.iloc[-1]['Low']]})

    # Use the trained model to predict the stock price for Tomorrow
    prediction = best_model.predict(future_data[features])
    print(f'Predicted Stock Price for Tomorrow: ${prediction[0]:.2f}')

except KeyError as e:
    print(f'Error accessing data for Tomorrow: {e}')

# Step 7: Plot the results with axis labels and hover functionality
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
