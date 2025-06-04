# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES



# Date: 08/04/2025
### NAME: ROSHINI S
### REG NO :212223240142
### AIM:
To implement ARMA model in python.
### ALGORITHM:
Import necessary libraries.   
Set up matplotlib settings for figure size.   
Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000    
data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits   .

Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using plot_acf and plot_pacf.    
Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000    
data points using the ArmaProcess class. Plot the generated time series and set the title and x- axis limits.   

Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using plot_acf and plot_pacf.   
### PROGRAM:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# Load the rainfall dataset
data = pd.read_csv('rainfall.csv')
# Use the 'rainfall' column; drop missing values
X = data['rainfall'].dropna()
# Set figure size
N = 1000
plt.rcParams['figure.figsize'] = [12, 6]
# Plot original data
plt.plot(X)
plt.title('Original Data - Rainfall')
plt.xlabel('Index')
plt.ylabel('Rainfall')
plt.show()
# Plot ACF and PACF of original data
plt.subplot(2, 1, 1)
plot_acf(X, lags=40, ax=plt.gca())
plt.title('ACF of Original Rainfall Data (Lags=40)')
plt.subplot(2, 1, 2)
plot_pacf(X, lags=25, ax=plt.gca())
plt.title('PACF of Original Rainfall Data (Lags=25)')
plt.tight_layout()
plt.show()
# Fit ARMA(1,1) model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
phi1_arma11 = arma11_model.params['ar.L1']
theta1_arma11 = arma11_model.params['ma.L1']
# Simulate ARMA(1,1)
ar1 = np.array([1, -phi1_arma11])
ma1 = np.array([1, theta1_arma11])
ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
plt.plot(ARMA_1)
plt.title('Simulated ARMA(1,1) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_1, lags=40)
plt.title('ACF of Simulated ARMA(1,1)')
plt.show()
plot_pacf(ARMA_1, lags=40)
plt.title('PACF of Simulated ARMA(1,1)')
plt.show()
# Fit ARMA(2,2) model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
phi1_arma22 = arma22_model.params['ar.L1']
phi2_arma22 = arma22_model.params['ar.L2']
theta1_arma22 = arma22_model.params['ma.L1']
theta2_arma22 = arma22_model.params['ma.L2']
# Simulate ARMA(2,2)
ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
ma2 = np.array([1, theta1_arma22, theta2_arma22])
ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N * 10)
plt.plot(ARMA_2)
plt.title('Simulated ARMA(2,2) Process')
plt.xlim([0, 500])
plt.show()
plot_acf(ARMA_2, lags=40)
plt.title('ACF of Simulated ARMA(2,2)')
plt.show()
plot_pacf(ARMA_2, lags=40)
plt.title('PACF of Simulated ARMA(2,2)')
plt.show()
```

### OUTPUT:
#### ORIGINAL DATA:
![image](https://github.com/user-attachments/assets/bdd4596b-acfa-48e9-ae1e-7752c94acf1a)

#### ACF ORIGINAL DATA:
![image](https://github.com/user-attachments/assets/c13b3063-a4a8-48d2-aab7-d50c26a6a793)
#### PACF ORIGINAL DATA:
![image](https://github.com/user-attachments/assets/0c11891e-1fb8-4222-88ae-987fdecd637c)




#### SIMULATED ARMA(1,1) PROCESS:
![image](https://github.com/user-attachments/assets/50f93f04-42e5-40ce-a379-44c3ca55405e)



##### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/7a5f394b-bfb7-4dca-989c-bded99797dc9)


#### Autocorrelation
![image](https://github.com/user-attachments/assets/88c6d453-5aea-4f2e-8a9f-8ee091db5728)



#### SIMULATED ARMA(2,2) PROCESS:
![image](https://github.com/user-attachments/assets/8c5c16c5-e4e7-45f6-9cae-f50bf6f8b722)


#### Partial Autocorrelation
![image](https://github.com/user-attachments/assets/09bff054-c794-4537-90bd-7bce9ab2e9d6)


#### Autocorrelation
![image](https://github.com/user-attachments/assets/c6ffb38a-f64b-4d01-a779-c84d8368cd0a)

### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
