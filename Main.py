import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.matrix import heatmap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('prices.csv')
sns.histplot(df['price'])
plt.show()
sns.heatmap(df.corr(), annot=True)
plt.show()

df.columns
x = df[['lot_area', 'living_area', 'num_floors',
        'num_bedrooms', 'num_bathrooms', 'waterfront',
        'year_built', 'year_renovated']]

y = df[['price']]
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lm = LinearRegression()
lm.fit(x_train, y_train)
print(lm.intercept_)
lm.coef_

x_train.columns
cdf = pd.DataFrame(data=lm.coef_.reshape(8, 1), index=x_train.columns, columns=['Coeff'])
predictions = lm.predict(x_test)
plt.scatter(y_test, predictions)
plt.show()
sns.histplot((y_test - predictions))
plt.show()

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))