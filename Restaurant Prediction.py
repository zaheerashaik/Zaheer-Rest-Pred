# Import Packages
import scipy
print ('scipy: %s' % scipy.__version__)
import numpy
print('numpy: %s' % numpy.__version__)
import matplotlib
print('matplotlib: %s' % matplotlib.__version__)
import pandas
print('pandas: %s' % pandas.__version__)
import sklearn
print('sklearn: %s' % sklearn.__version__)
import statsmodels
print('statsmodels: %s' % statsmodels.__version__)


# Read the datasets
import pandas as pd
series = pd.read_csv('Item Wise.csv')
#series = series.drop([0,1,2,3,4,5])
series = series.drop(series.columns[4],axis=1)
series.columns = ['Item', 'Date', 'Quantity', 'Amount']
series = series.loc[series['Item'] == 'Chicken Biryani (Regular)']
series.tail()


# Split the Training & Testing files
series = series[['Date', 'Quantity']]
series.head(10)
split_point = len(series)-10
dataset, validation = series[0:split_point], series[split_point:]
print('Dataset %d, Validation %d' % (len(dataset), len(validation)))
dataset.to_csv('dataset.csv')
validation.to_csv('validation.csv')


# Exploring the Training (Dataset.csv) file
import seaborn as sns
series.describe()


# Import more libraries
from pandas import Series
from matplotlib import pyplot
import matplotlib.pylab as plt
import numpy as np
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10

# Convert the Data Types
series['Quantity'] = pd.to_numeric(series['Quantity'], errors='coerce')
IndexedSeries = series.set_index('Quantity')
series['Date'] = pd.to_datetime(series['Date'], infer_datetime_format=True)
IndexedSeries = series.set_index(['Date'])
IndexedSeries.tail()

# Plot the time series chart
plt.xlabel("Date")
plt.ylabel("Quantity")
plt.plot(IndexedSeries)

# Saving the plot
plt.savefig('Time Series.png')

# Rolling Statistics
rolmean = IndexedSeries.rolling(window=365).mean()
rolstd = IndexedSeries.rolling(window=365).std()
orig = plt.plot(IndexedSeries, color='blue', label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='yellow', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

# Dickey-Fuller Test
from statsmodels.tsa.stattools import adfuller
print('Results of Dickey-Fuller Test:')
dftest = adfuller(IndexedSeries['Quantity'], autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)

# Log Scale of the Data
IndexedSeries_logscale = np.log(IndexedSeries)
plt.plot(IndexedSeries_logscale) 
movingAverage = IndexedSeries_logscale.rolling(window=365).mean()
movingSTD = IndexedSeries_logscale.rolling(window=365).std()
plt.plot(IndexedSeries_logscale)
plt.plot(movingAverage, color='red')

# Making the Data stationary
datasetLogScaleMinusMovingAverage = IndexedSeries_logscale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.tail(10)

# Dickey-Fuller Test Complete Code
from statsmodels.tsa.stattools import adfuller
def test_stationarity(timeseries):
    
    # Determinint rolling statistics
    movingAverage = timeseries.rolling(window=365).mean()
    movingSTD = timeseries.rolling(window=365).std()
    
    # Plot rolling statistics
    orig = plt.plot(timeseries, color='green', label='Original')
    mean = plt.plot(movingAverage, color='red', label='Rolling Mean')
    std = plt.plot(movingSTD, color='yellow', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    plt.savefig('ADF Test.png')

    # Perform Dickey-Fuller Test
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries['Quantity'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' %key] = value
    print(dfoutput)

test_stationarity(datasetLogScaleMinusMovingAverage)



# Log Rolling Statistics
rollogmean = IndexedSeries_logscale.rolling(window=365).mean()
rollogstd = IndexedSeries_logscale.rolling(window=365).std()
orig = plt.plot(IndexedSeries_logscale, color='blue', label='Original')
mean = plt.plot(rollogmean, color='red', label='Rolling Mean')
std = plt.plot(rollogstd, color='yellow', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)

# Dickey-Fuller Test Again
from statsmodels.tsa.stattools import adfuller
print('Results of Log Data Dickey-Fuller Test:')
dflogtest = adfuller(IndexedSeries_logscale['Quantity'], autolag='AIC')
dflogoutput = pd.Series(dflogtest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
for key, value in dflogtest[4].items():
    dflogoutput['Critical Value (%s)'%key] = value
print(dflogoutput)

# Weighted Average
exponentialDecayWeightedAverage = IndexedSeries_logscale.ewm(halflife=365, min_periods=0, adjust=True).mean()
plt.plot(IndexedSeries_logscale)
plt.plot(exponentialDecayWeightedAverage, color='red')

# Log Scale Weighted Average
datasetLogScaleMinusMovingExponentialDecayAverage = IndexedSeries_logscale - exponentialDecayWeightedAverage
test_stationarity(datasetLogScaleMinusMovingExponentialDecayAverage)

# Shifting the Average
datasetLogDiffShifting = IndexedSeries_logscale - IndexedSeries_logscale.shift()
plt.plot(datasetLogDiffShifting)
datasetLogDiffShifting.dropna(inplace=True)
test_stationarity(datasetLogDiffShifting)

# Separating Components
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(IndexedSeries_logscale, freq=7)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.subplot(411)
plt.plot(IndexedSeries_logscale, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout

# Saving the decomposed charts
plt.savefig('Decomposition.png')

# ADF Test for Residuals
decomposedLogData = residual
decomposedLogData.dropna(inplace=True)
test_stationarity(decomposedLogData)

# ACF & PACF Plots
from statsmodels.tsa.stattools import acf, pacf
lag_acf = acf(datasetLogDiffShifting, nlags=20)
lag_pacf = pacf(datasetLogDiffShifting, nlags=20, method='ols')

# Plot ACF 
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='green')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='green')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='green')
plt.title('Autocorrelation Function')

# Plot PACF 
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='green')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='green')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffShifting)), linestyle='--', color='green')
plt.title('Partial Autocorrelation Function')

# Display and save the ACF & PACF charts
plt.tight_layout()
plt.savefig('ACF & PACF.png')

# Import ARIMA Model
from statsmodels.tsa.arima_model import ARIMA

# AR Model
model = ARIMA(IndexedSeries_logscale, order=(7,1,5))
results_AR = model.fit(disp=-1, transparams=True)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color='red')
plt.title('RSS: %.4f' %sum((results_AR.fittedvalues-datasetLogDiffShifting['Quantity'])**2))
print('Plotting AR Model')

# MA Model
model = ARIMA(IndexedSeries_logscale, order=(7,1,5))
results_MA = model.fit(disp=-1, transparams=True)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color='red')
plt.title('RSS: %.4f' %sum((results_MA.fittedvalues- datasetLogDiffShifting['Quantity'])**2))
print('Plotting MA Model')

# ARIMA Model
model = ARIMA(IndexedSeries_logscale, order=(7,1,5))
results_ARIMA = model.fit(disp=-1, transparams=True)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color='red')
plt.title('RSS: %.4f' %sum((results_ARIMA.fittedvalues- datasetLogDiffShifting['Quantity'])**2))
print('Plotting MA Model')

# Predictions 
predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)
print(predictions_ARIMA_diff.head())

# Convert to cumulative sum
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())

predictions_ARIMA_log = pd.Series(IndexedSeries_logscale['Quantity'].ix[0], index=IndexedSeries_logscale.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA_log.head()

predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(IndexedSeries)
plt.plot(predictions_ARIMA)
IndexedSeries_logscale

# Checking existing rows
IndexedSeries

# Making the predictions
results_ARIMA.plot_predict(1,579)
x = results_ARIMA.forecast(steps=30)
plt.savefig('Predictions.png')

x[0]
len(x[0])
np.exp(x[0])