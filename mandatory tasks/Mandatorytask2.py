import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("train_data.csv")

print("First few rows of the dataset:")
print(df.head())

print("\nSome quick info about the dataset:")
print(df.info())

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.scatter(df['w'], df['y'], color='orange')
plt.xlabel('w')
plt.ylabel('y')
plt.title('w vs y')

plt.subplot(1,2,2)
plt.scatter(df['x'], df['y'], color='green')
plt.xlabel('x')
plt.ylabel('y')
plt.title('x vs y')
plt.tight_layout()
plt.show()

X = df[['w', 'x']]
y = df['y']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel('Actual y')
plt.ylabel('Predicted y')
plt.title('Actual vs Predicted y')
plt.grid(True)
plt.show()

mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")


