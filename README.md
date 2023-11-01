# Ex.No.8-ARIMA-model-in-python
## AIM:
To Implementation of ARIMA Model Using Python.

## Procedure:
1.Import necessary libraries

2.Read the CSV file,Display the shape and the first 20 rows of the dataset

3.Set the figure size for plots

4.Import the SARIMAXfrom statsmodels.tsa.statespace.sarimax

5.Calculate root mean squared error

6.Calculate mean squared error

## Program:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib import pyplot
df=pd.read_csv("/content/TEMP.csv",index_col=0,parse_dates=True)
data1=df
df.pop("JAN")
df.pop("FEB")
df.pop("MAR")
df.pop("APR")
df.pop("MAY")
df.pop("JUN")
df.pop("JUL")
df.pop("AUG")
df.pop("SEP")
df.pop("OCT")
df.pop("NOV")
df.pop("DEC")
df.pop("JAN-FEB")
df.pop("MAR-MAY")
df.pop("OCT-DEC")
df.pop("JUN-SEP")
df.shape
df.head()
x=df.values
x
df.plot(figsize=(10,5))
from statsmodels.tsa.stattools import adfuller

dftest= adfuller(df['ANNUAL'],autolag='AIC')
print("1. ADF : ",dftest[0])
print("2. P-Value : ",dftest[1])
print("3. Number Of Lags : ",dftest[2])
print("4.Num of observation used FOr ADF Regression  and Critical value Calculation :",dftest[3])
for key,val in dftest[4].items():
     print("\t",key, ":",val)
from pmdarima import auto_arima
import warnings
warnings.filterwarnings("ignore")
stepwise_fit = auto_arima(df,trace=True,suppress_warnings=True)
stepwise_fit.summary()
train=x[:len(df)-12]
test=x[len(df)-12:]
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train,order = (0, 1, 1),seasonal_order =(2, 1, 1, 12))

result = model.fit()
result.summary()
start=len(train)
end=len(train)+len(test) -1
pred=result.predict(start,end,type='levels')
pred
plt.plot(pred)
plt.plot(test)
from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse

# Calculate root mean squared error
rmse(test, pred)

# Calculate mean squared error
mean_squared_error(test, pred)
pred
```
## Output:
### df.shape
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/0193f547-1216-4838-816c-7bd7d4352ef6)


### df.head()
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/af84dccd-c100-4ff0-b241-46a3997405a3)


### x values
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/621d10c8-e77f-46d7-a680-493ddd2f4f82)


### df.plot()
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/5081a9f8-5d57-44ea-bb6d-31bc2812c0d2)


### key,val
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/df0f4379-f958-4f66-b981-40302ff8ecb2)


### stepwise_fit.summary()
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/dec13cdf-951d-4d64-b5e8-9d06935cbac6)


### result.summary()
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/47c3671d-b588-43ed-a1b9-233621d04f45)


### pred
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/a8b4376a-408a-4511-a72f-98f18ecb3bfd)


### plt.plot(pred),plt.plot(test)
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/bae111f7-dfca-4db2-8568-50504efbe479)

### Calculate mean squared error
![image](https://github.com/s-adhithya/Ex.No.8-ARIMA-model-in-python/assets/113497423/726e12d0-41f8-4d07-b490-37341891bf47)


## Result:
Thus we have successfully implemented the ARIMA Model using above mentioned program.

