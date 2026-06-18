import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("house.csv")

X = df[['area',
        'bedrooms',
        'bathrooms',
        'kitchen',
        'storeroom',
        'bighall',
        'parking',
        'garden',
        'swimmingpool']]

y = df['price']

model = LinearRegression()

model.fit(X, y)