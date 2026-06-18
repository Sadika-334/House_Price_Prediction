import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load data
df = pd.read_csv("house.csv")

# Features and target
X = df[['area']]
y = df['price']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict custom value
new_house = pd.DataFrame({'area':[2000]})
price = model.predict(new_house)

print("Predicted Price:", price[0])
import matplotlib.pyplot as plt

plt.scatter(X, y)
plt.plot(X, model.predict(X))

plt.xlabel("Area")
plt.ylabel("Price")
plt.title("House Price Prediction")

plt.show()