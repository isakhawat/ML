#!/usr/bin/env python
# coding: utf-8

#   
#  

#  
# ## Check out the data
# We've been able to get some data from your neighbor for housing prices as a csv set, let's get our environment ready with the libraries we'll need and then import the data!
# ### Import Libraries

 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Check out the Data

 


diabetes = pd.read_csv('C:\\Users ........ \\diabetes.csv')

diabetes.head()


diabetes.info()


diabetes.describe()

diabetes.columns


# # EDA
# 
# Let's create some simple plots to check out the data!



sns.pairplot(diabetes)

diabetes.columns

sns.distplot(diabetes['Outcome'])

sns.distplot(diabetes['Age'])

sns.heatmap(diabetes.corr())


Training a Linear Regression Model
# 
# Let's now begin to train out regression model! We will need to first split up our data into an X array that contains the features to train on, and a y array with the target variable, in this case the Price column. We will toss out the Address column because it only has text info that the linear regression model can't use.

X = diabetes[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age']]
y = diabetes['Outcome']


# ## Train Test Split
# Now let's split the data into a training set and a testing set. We will train out model on the training set and then use the test set to evaluate the model.




from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# ## Creating and Training the Model

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)


# ## Model Evaluation
# 
# Let's evaluate the model by checking out it's coefficients and how we can interpret them.


# print the intercept
print(lm.intercept_)

coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df

# ## Predictions from our Model
# Let's grab predictions off our test set and see how well it did!

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)


# Residual Histogram

sns.distplot((y_test-predictions),bins=50);


# ## Regression Evaluation Metrics
# Here are three common evaluation metrics for regression problems:
# 
# **Mean Absolute Error** (MAE) is the mean of the absolute value of the errors:
# 
# $$\frac 1n\sum_{i=1}^n|y_i-\hat{y}_i|$$
# 
# **Mean Squared Error** (MSE) is the mean of the squared errors:
# 
# $$\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2$$
# 
# **Root Mean Squared Error** (RMSE) is the square root of the mean of the squared errors:
# 
# $$\sqrt{\frac 1n\sum_{i=1}^n(y_i-\hat{y}_i)^2}$$
# 
# Comparing these metrics:
# 
# - **MAE** is the easiest to understand, because it's the average error.
# - **MSE** is more popular than MAE, because MSE "punishes" larger errors, which tends to be useful in the real world.
# - **RMSE** is even more popular than MSE, because RMSE is interpretable in the "y" units.
# 
# All of these are **loss functions**, because we want to minimize them.

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

