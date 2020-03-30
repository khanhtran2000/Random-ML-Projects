import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
%matplotlib inline

# Import the csv file
url = 'https://raw.githubusercontent.com/khanhtran2000/random_projects/master/corona%20us.csv'
dataset = pd.read_csv(url)

# Use the log of 'US cases' column
dataset['log_US cases'] = np.log(dataset['US cases'])

# Setting training and testing sets
X_train = dataset.Time[31:50].values.reshape(-1,1)
y_train = dataset['log_US cases'][31:50].values.reshape(-1,1)
X_test = dataset.Time[50:].values.reshape(-1,1)
y_test = dataset['log_US cases'][50:].values.reshape(-1,1)

regressor = LinearRegression()
regressor.fit(X_train, y_train) #training algorithm
y_pred = regressor.predict(X_test)

# Run regressor.intercept_ to retrieve the intercept, which is -4.313
# regressor.coef_ to get the slope, which is 0.286
intercept = -4.313
slope = 0.286

# Print out metrics
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('R-Squared Score:', r2_score(y_test, y_pred))

# Print out intercept and slope
print('Intercept:', intercept)
print('Slope:', slope)

# Plotting the testing set
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('Time (days from 01/31/2020)')
plt.ylabel('log(US cases)')
plt.show()

# Adding new dataset for prediction
add_dataset = pd.DataFrame(
    {
        'Date': [
                    '3/27/2020', '3/28/2020', '3/29/2020', '3/30/2020', '3/31/2020',
                    '4/1/2020', '4/2/2020', '4/3/2020', '4/4/2020', '4/5/2020', '4/6/2020', '4/7/2020',
                    '4/8/2020', '4/9/2020'
                ],
        'Time': [i for i in range(56,70)],
        'US cases': '',
        'log_US cases': ''
    }
)

# Calculate the log_US cases for the new dataset
add_dataset['log_US cases'] = add_dataset.Time*slope + intercept

# Calculate the US cases for the new dataset
add_dataset['US cases'] = np.e**add_dataset['log_US cases']

cases_pred = regressor.predict(add_dataset['Time'].values.reshape(-1,1))

# Plotting the new data
plt.scatter(add_dataset['Time'], add_dataset['log_US cases'], color='gray')
plt.plot(add_dataset['Time'], cases_pred, color='red', linewidth=2)
plt.xlabel('Time (days from 01/31/2020)')
plt.ylabel('log(US cases)')
plt.show()


dataset = pd.concat([dataset, add_dataset], ignore_index=True)

# Final plotting 
plt.scatter(dataset['Time'], dataset['US cases'], color='gray')
plt.xlabel('Time (days from 01/31/2020)')
plt.ylabel('US Cases')
plt.subplots_adjust(wspace=0.6, hspace=0.6, left=0.15, bottom=0.1, right=0.96, top=0.96)
plt.show()

