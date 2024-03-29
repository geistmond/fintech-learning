# Random Forest Classifier

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime
import os
#import yfinance

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import precision_score

# Load data about a stock
company = 'AAPL'

from api_keys import av_key
av_api_key = av_key
#av_api_key = os.getenv('$ALPHAVANTAGE_API_KEY')


# Load and prepare training data
start = datetime.datetime(2010,1,1)
end = datetime.datetime(2020,1,1)
data = web.DataReader(company, "av-daily", start, end, api_key=av_api_key)
#print(data)

# Put result in CSV file
data.to_csv(company+'.csv', index=True)
csv_data = pd.read_csv(company+'.csv')

# Add data about next day values
data['tomorrow'] = data['close'].shift(-1)
data['target'] = (data['tomorrow'] > data['close']).astype(int) # 1 if upward delta between closing prices each day, 0 if not

# Load and prepare test data
test_start = datetime.datetime(2020,1,1)
test_end = datetime.datetime.now()
test_data = web.DataReader(company, 'av-daily', test_start, test_end, api_key=av_api_key)
test_data['tomorrow'] = test_data['close'].shift(-1)
test_data['target'] = (test_data['tomorrow'] > test_data['close']).astype(int)

# Build model for Classification
classifier = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
predictors = ['close', 'volume', 'open', 'high', 'low']
classifier.fit(data[predictors], data['target'])

total_dataset = pd.concat((data, test_data), axis=0)

actual_prices = test_data['close'].values
predicted_prices = classifier.predict(test_data[predictors])

def predict_class(data, test_data, predictors, classifier):
    classifier.fit(data[predictors], data['target'])
    preds = classifier.predict(test_data[predictors])
    preds = pd.Series(preds, index=test_data.index, name='predictions')
    combined = pd.concat([test_data['target'], preds], axis=0)
    #prec = precision_score(test_data['target'], preds)
    return combined

def backtest(data, classifier, predictors, start=2500, step=250):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict_class(train, test, predictors, classifier)
        all_predictions.append(predictions)
    
    return pd.concat(all_predictions)

classifier_predictions = backtest(total_dataset, classifier, predictors)

#print(total_dataset) # includes classifier predictions

# Train Regressor on data that has been pre-classified.
regressor = RandomForestRegressor()
regressor_predictors = ['volume', 'open', 'high', 'low', 'target']
regressor.fit(total_dataset[regressor_predictors], total_dataset['close'])

predictions = regressor.predict(total_dataset[regressor_predictors])
#print(predictions)

original_values = total_dataset['close'] # exclude last value which is NaN
predicted_values = pd.Series(predictions, index=total_dataset.index)

print(original_values)
print(predicted_values)

# Plot the test predictions
plt.plot(total_dataset['close'], color="black", label=f"Actual {company} price")
plt.plot(predictions, color="green", label=f"Predicted {company} price")
plt.title(f"{company} Share Price")
plt.xlabel('Time')
plt.ylabel(f'{company} Share Price')
plt.legend()
plt.show()