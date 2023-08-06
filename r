import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("Data/house_rental_data.csv.txt")

X = data.drop("Price", axis=1)
y = data["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

new_house_features = pd.DataFrame({
    "Sqft": [1500],
    "Floor": [4],
    "TotalFloor": [10],
    "Bedroom": [2],
    "Living.Room": [1],
    "Bathroom": [1]
})

predicted_rent = model.predict(new_house_features)
print("Predicted Rent:", predicted_rent[0])
