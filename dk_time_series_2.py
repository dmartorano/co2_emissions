#!/usr/bin/env python
# coding: utf-8

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

get_ipython().run_line_magic('matplotlib', 'inline')

# This will allow us to avoid a FutureWarning when plotting.
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[43]:


df = pd.read_csv('../datasets/USAverage_Energy_CO2_1970to2021.csv')


# In[44]:


#checking what we got here 
df.head()


# In[45]:


# setting year as the index 
df.set_index('Year', inplace=True)


# In[46]:


# Convert the index to a DatetimeIndex because the 
# that's the way seasonal_decompose wants it
df.index = pd.to_datetime(df.index, format='%Y')


# In[47]:


# double checking 
df.head()


# In[48]:


df.shape


# In[49]:


df.dtypes


# # Linear Time Series Modeling

# In[50]:


def plot_series(df, cols=None, title='Title', xlab=None, ylab=None):
    
    # Set figure size to be (18, 9).
    plt.figure(figsize=(18,9))
    
    # Iterate through each column name.
    for col in cols:
        
        # Generate a line plot of the column name.
        # You only have to specify Y, since our
        # index will be a datetime index.
        plt.plot(df[col])
        
    # Generate title and labels.
    plt.title(title, fontsize=26)
    plt.xlabel(xlab, fontsize=20)
    plt.ylabel(ylab, fontsize=20)
    
    # Enlarge tick marks.
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18);


# In[51]:


# looking at the changes in Co2 emission
plot_series(df, cols = ['CO2_Value'], title = "Change in CO2 Emission in United States", xlab="Year", ylab="CO2 Emission")


# # Interpretation
# ____
# 
# Based on this plot, the general trend for CO2 emission is going down from 1970 to 2020. Starting from roughly 2007, the CO2 emission appears to be going down. 

# In[52]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[67]:


# Decompose time series into trend, seasonal, and residual components.
decomp = seasonal_decompose(df['CO2_Value'])
 
# Plot the decomposed time series.
decomp.plot();


# # CO2 Interpretation
# _____
# 
# For the trend plot, we observe the general trend of CO2 emission going down from 1970 to 2021. However, between 1990 and 2000, there is a slight increase in the CO2 emission before it gradually comes down from early 2000s and onward. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.

# In[68]:


# Decompose time series into trend, seasonal, and residual components for Coal Value
decomp = seasonal_decompose(df['Coal_Value'])
 
# Plot the decomposed time series.
decomp.plot();


# # Coal Interpretation
# _____
# 
# For the trend plot, we observe the general trend of coal energy consumption going down from 1970 to 2021. However, between 1990 and 2000, there is a slight increase in the CO2 emission before it gradually comes down around 2009 and onward. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.
# 
# To note, the coal trend plot seems to follow similiar trend like that of the CO2 trend plot. 

# In[69]:


# Decompose time series into trend, seasonal, and residual components for Coal Value
decomp = seasonal_decompose(df['Natural_Gas_Value'])
 
# Plot the decomposed time series.
decomp.plot();


# # Natural Gas Interpretation
# _______
# 
# For the trend plot, we observe the general trend of natural gas energy consumption going up from 1970 to 2021. However, we observe the natural gas at its lowest point in 1985 before gradually increasing afterwards. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.
# 

# In[70]:


# Decompose time series into trend, seasonal, and residual components for Nuclear Energy Value
decomp = seasonal_decompose(df['Nuclear_Energy_Value'])
 
# Plot the decomposed time series.
decomp.plot();


# # Nuclear Energy Interpretation
# _______
# 
# For the trend plot, we observe an increase general trend of nuclear energy consumption from 1970 to 2021. However, we observe a slight dip in 1997 before the trend increases again afterwards. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.
# 

# In[71]:


# Decompose time series into trend, seasonal, and residual components for Petroleum Energy Value
decomp = seasonal_decompose(df['Petroleum_Energy_Value'])
 
# Plot the decomposed time series.
decomp.plot();


# # Petroleum Energy Interpretation
# ____
# 
# For the trend plot, we observe a general trend of petroleum energy consumption increasing from 1970 to 2021. However, we observe the petroleum energy consumption going down in 1977, going upw in 1980, and going back down in 1983. After that, the petroleum energy consumption trends gradually increases. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.
# 

# In[72]:


# Decompose time series into trend, seasonal, and residual components for Renewable Energy 
decomp = seasonal_decompose(df['Renewable_Energy_Value'])

# Plot the decomposed time series.
decomp.plot();


# # Renewable Energy  Interpretation
# _____
# 
# For the trend plot, we observe an increase in general trend of renewable energy consumption from 1970 to 2021. As for the seasonal trend, there doesn't seem to be any pattern and is at constant value of 0. For the residual trend, it appears that there is no pattern.

# # ACF and PACF plots

# In[75]:


# Generate an ACF plot of the CO2 data with 10 time periods.
# when you have seasonality, this is the type of pattern
# you should see in your plot
plot_acf(df['CO2_Value'], lags=10);


# In[76]:


# Generate an PACF plot of the CO2 data with 10 time periods.
# when you have seasonality, this is the type of pattern
# you should see in your plot
plot_pacf(df['CO2_Value'], lags=10);


# # CO2 ACF and PACF Plot Interpretations
# ______
# 
# As for ACF and PACF plots, we see small values having positive autocorrelations. Therefore there is a trend. On the other hand, for the PACF plot, we see some positive and some negative significant partial autocorrelations, which usually indicates strong seasonal fluctuations.

# In[77]:


# Create a column called `lag_1` that lags Passengers by one month.
df['lag_1'] = df["CO2_Value"].shift(1)

# Create a column called `lag_2` that lags Passengers by two months.
df['lag_3'] = df["CO2_Value"].shift(3)

# Create a variable called `time` that takes on a value of 0 in January 1970,
# then increases by 1 each year until the end of the dataframe.
df['time'] = range(0, df.shape[0])


# In[78]:


df.head()


# In[79]:


df.tail()


# # Train_Test Split
# ________
# 

# In[80]:


# Generate train/test split.
# when dealing with time series data, 
# make sure to make shuffle to False
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns =['CO2_Value']),
                                                    df['CO2_Value'],
                                                    test_size = 0.2, shuffle=False)


# In[81]:


# Check shape to confirm we did this properly.
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[82]:


# Import statsmodels.
import statsmodels.api as sm


# In[83]:


# Before fitting a model in statsmodels, what do we need
# to do? (Hint: Think intercept.)
# for beta 0 create a constant of all 1
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)



# Confirm.
X_train.head()


# In[84]:


# statsmodels won't be able to handle missing values.
X_train.dropna(inplace=True)

y_train = y_train[X_train.index]


# In[85]:


X_train.index


# In[86]:


# Remember that, in statsmodels, we pass our data 
# in when we instantiate the model!

lm = sm.OLS(y_train, X_train)


# In[87]:


# Then we fit our model.
lm_results = lm.fit()


# In[88]:


# Display our summary!
print(lm_results.summary())


# In[89]:


# Generate predicted test values.
lm_results.predict(X_test)


# In[90]:


# Import R2 score and MSE.
from sklearn.metrics import r2_score, mean_squared_error


# In[91]:


# Calculate R2 score.
r2_score(y_test, lm_results.predict(X_test))


# In[92]:


# Calculate RMSE.
mean_squared_error(y_test, lm_results.predict(X_test)) ** 0.5


# # Result Interpretations
# _____
# 
# The R2 score turns out as 0.45, indicating that the model moderately fits the data and has moderate predictive power for explaining CO2 emissions based on the provided predictor variables. However, there is still room for improvement, and additional analysis may be necessary to refine the model further. 
# 
# As for the cofficients for each variable, here are the following interpretations:
# 
# * Coal_Value (2.156e-05): For each unit increase in coal energy consumption, the model predicts a CO2 emission increase of approximately 2.156e-05 units, holding all other predictors constant.
# 
# * Natural_Gas_Value (9.811e-06): For each unit increase in natural gas energy consumption, the model predicts a CO2 emission increase of approximately 9.811e-06 units, holding all other predictors constant.
# 
# * Nuclear_Energy_Value (4.067e-05): For each unit increase in nuclear energy consumption, the model predicts a CO2 emission increase of approximately 4.067e-05 units, holding all other predictors constant.
# 
# * Petroleum_Energy_Value (1.027e-05): For each unit increase in petroleum energy consumption, the model predicts a CO2 emission increase of approximately 1.027e-05 units, holding all other predictors constant.
# 
# * Renewable_Energy_Value (-2.865e-06): For each unit increase in renewable energy consumption, the model predicts a decrease in CO2 emission of approximately -2.865e-06 units, holding all other predictors constant. This negative coefficient suggests that higher renewable energy consumption is associated with lower CO2 emissions.
# 
# * lag_1 (0.0037): This coefficient represents the effect of the lagged value of the target variable (CO2 emission) at time t-1 on the current CO2 emission. For each unit increase in the lagged CO2 emission, the current CO2 emission is predicted to increase by approximately 0.0037 units.
# 
# * lag_3 (-0.1538): This coefficient represents the effect of the lagged value of the target variable (CO2 emission) at time t-3 on the current CO2 emission. For each unit increase in the CO2 emission three time steps ago, the current CO2 emission is predicted to decrease by approximately 0.1538 units.
# 
# * time (-0.2475): This coefficient represents the effect of time (assuming it's a continuous variable) on CO2 emission. For each unit increase in time, the model predicts a decrease in CO2 emission of approximately 0.2475 units, holding all other predictors constant.
# 

# In[93]:


# Let's plot our predictions! 

# Set figure size.
plt.figure(figsize=(20,10))

# Plot training data.
plt.plot(y_train.index, y_train.values, color = 'blue')

# Plot testing data.
plt.plot(y_test.index, y_test.values, color = 'orange')

# Plot predicted test values.
plt.plot(lm_results.predict(X_test), color = 'green')

# Set label.
plt.title(label = 'Forecasting CO2 Emissions from US States by 1970 to 2021', fontsize=24)
plt.xlabel(xlabel ='Year', fontsize=20)
plt.ylabel(ylabel = 'CO2 Emission', fontsize=20)


# Resize tick marks.
plt.xticks(fontsize=20)
plt.yticks(fontsize=20);


# # Interpretation 
# _____
# 
# For this graph, the orange line represents the predicted CO2 emission while the green line represents the predicted values from all of the energy sources: coal, natural gas, nuclear, petroleum, and renewable energy. Based on this graph, we observe in this model that the CO2 emission is predicted to go down.
