import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import csv
import os
from crypto_predictor.crypto_predictor.settings import BASE_DIR
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential



def history_comparison(crypto, fiat_currency, prediction_days):

    #This function predicts one day into the future based on trends of the previous 60 days, beginning from the "start" variable
    #Then compares the prediction to the actual price

    #Initialising data
    start = dt.datetime(2016,1,1)
    end = dt.datetime.now()
    df = web.DataReader(f'{crypto}-{fiat_currency}', 'yahoo', start, end)
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_df = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    #Creating test data sets
    x_train, y_train = [], []

    for x in range(prediction_days, len(scaled_df)):
        x_train.append(scaled_df[x-prediction_days:x, 0])
        y_train.append(scaled_df[x, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Creating Neural Network
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    # Testing in the model
    test_start = dt.datetime(2020,1,1)
    test_end = dt.datetime.now()
    test_data = web.DataReader(f'{crypto}-{fiat_currency}', 'yahoo', test_start, test_end)
    actual_prices = test_data['Close'].values

    total_df = pd.concat((df['Close'], test_data['Close']), axis=0)

    model_inputs = total_df[len(total_df) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.fit_transform(model_inputs)

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predicted_prices = model.predict(x_test)
    predicted_prices = scaler.inverse_transform(predicted_prices)

    plt.plot(actual_prices, color='black', label='Actual Prices')
    plt.plot(predicted_prices, color='red', label='Predicted Prices')
    plt.title(f'{crypto} price prediction by machine learning')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(BASE_DIR, 'frontend/static/images/plot.png'))
    plt.show()


    #Predicting one day into the future based on learned data above
    learned_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs) + 1, 0]]
    learned_data = np.array(learned_data)
    learned_data = np.reshape(learned_data, (learned_data.shape[0], learned_data.shape[1], 1))

    oneday_prediction = model.predict(learned_data)
    oneday_prediction = scaler.inverse_transform(oneday_prediction)
    print()



def price_prediction(crypto, fiat_currency, prediction_days):

    #This function predicts the next 14 days based on the trends since the "start" variable

    #Initialising data
    start = dt.datetime(2020,1,1)
    end = dt.datetime.now()
    df = web.DataReader(f'{crypto}-{fiat_currency}', 'yahoo', start, end)
    df['Prediction'] = df[['Close']].shift(-prediction_days)

    #Creating an independent and dependent data set and removing last 14 rows of null data
    x = np.array(df[['Close']])
    x = x[:- prediction_days]

    y = df['Prediction'].values
    y = y[:- prediction_days]

    #Splitting data into 0.8 training and 0.2 testing data sets, then training model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
    linReg = LinearRegression()
    linReg.fit(x_train, y_train)

    #Testing model using score and creating projection variable
    conf = linReg.score(x_test, y_test)
    print('Linear Regression Confidence:', conf)
    x_projection = np.array(df[['Close']])[-prediction_days:]

    #Print linear regression models predictions for last 14 days, write to CSV
    lr_prediction = linReg.predict(x_projection)
    print(lr_prediction)
    wtr = csv.writer(open ('predictions.csv', 'w'), delimiter=',', lineterminator='\n')
    for x in lr_prediction : wtr.writerow ([x])

    


history_comparison('ETH', 'USD', 60)
#price_prediction('ETH', 'USD', 14)