import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import mplcursors  # Import mplcursors for hover functionality
import numpy as np


# Step 1: Download the data
ticker = 'NVDA'
start_date = '2020-01-01'
end_date = '2024-06-14'
data = yf.download(ticker, start=start_date, end=end_date)

# Step 2: Preprocess the data
data.dropna(inplace=True)
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Step 3: Split the data
features = ['Close', 'MA20', 'MA50']
X = data[features]
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Build the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Step 6: Predict the stock price for Tomorrow
try:
    # Create a DataFrame with data for Tomorrow
    future_date = '2024-07-17'
    future_data = pd.DataFrame({'Close': [data.iloc[-1]['Close']],
                                'MA20': [data.iloc[-1]['MA20']],
                                'MA50': [data.iloc[-1]['MA50']]})


    # Use the trained model to predict the stock price for Tomorrow
    prediction = model.predict(future_data[features])
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
