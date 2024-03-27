import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import os

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load data about a stock
company = 'AAPL'

from api_keys import av_key
av_api_key = av_key
#av_api_key = os.getenv('$ALPHAVANTAGE_API_KEY')

start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,1,1)

#data_old = web.DataReader(company, 'yahoo', start, end)

data = web.DataReader(company, "av-daily", start, end, api_key=av_api_key)

print(data)

# Prepare data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data['close'].values.reshape(-1,1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1))) # Input
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True)) # Connect middle layer for input handling
model.add(Dropout(0.2))
model.add(LSTM(units=50)) # Connect middle layer to reduce
model.add(Dropout(0.2))
model.add(Dense(units=1)) # Prediction output from NN of the next closing price from Dense layer

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32)

# Load test data
test_start = datetime.datetime(2020,1,1)
test_end = datetime.datetime.now()

test_data = web.DataReader(company, 'av-daily', test_start, test_end, api_key=av_api_key)
actual_prices = test_data['close'].values

total_dataset = pd.concat((data['close'], test_data['close']), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make predictions on test data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the test predictions
plt.plot(actual_prices, color="black", label=f"Actual {company} price")
plt.plot(predicted_prices, color="green", label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()