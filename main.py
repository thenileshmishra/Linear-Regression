import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv("salary_prediction_data.csv")


education_mapping = {
  'High School' : 1,
  'Bachelor' : 2,
  'Master':3,
  'PhD':4
}

data['Education_Encoding'] = data["Education"].map(education_mapping)


Job_title_mapping = {
    "Analyst" :1,
    "Engineer":2,
    "Manager":3,
    "Director":4
}

data["Job_title_Encoding"] = data["Job_Title"].map(Job_title_mapping)

Gender_mapping ={
    "Male" :1,
    "Female":2
}

data["Sex"] = data["Gender"].map(Gender_mapping)

loc_mapping = {
    "Rural":1,
    "Suburban":2,
    "Urban":3
}

data["locations"] = data["Location"].map(loc_mapping)

X = data.drop(columns = ["Education", "Gender", "Location", "Job_Title", "Salary"])
y = data["Salary"].values

# Train-test Split function
def split_data(X,y, split_ratio=0.8):
    # Calculate split index
    split_index = int(len(data) * split_ratio)

    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]

    return X_train, X_test, y_train, y_test

# Linear Regression implementation
class LinearRegression:

    def __init__(self, learning_rate=0.01, epoch=1000):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None

    # Inside the LinearRegression class

    def fit(self, X, y):
        # Standardize the data
        self.x_mean = np.mean(X)
        self.x_std = np.std(X)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)

          # Avoid division by zero
        self.x_std[self.x_std == 0] = 1

        X = (X - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        if X.ndim == 1:  # Normalize single feature
            X = X.reshape(-1, 1)

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0   

        max_grad = 1.0  # Gradient clipping threshold

        for _ in range(self.epoch):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Apply gradient clipping
            dw = np.clip(dw, -max_grad, max_grad)
            db = np.clip(db, -max_grad, max_grad)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        # Standardize features using training mean and std
        X = (X - self.x_mean) / self.x_std
        y_pred_standardized = np.dot(X, self.weights) + self.bias
        # Unstandardize predictions
        y_pred = y_pred_standardized * self.y_std + self.y_mean
        return y_pred

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        return np.mean((y_pred - y) ** 2)  # Mean Squared Error

# Split the data
X_train, X_test, y_train, y_test = split_data(X, y, split_ratio=0.8)

# Train the model
model = LinearRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
mse = model.evaluate(X_test, y_test)

print(f"Mean Squared Error: {mse:.2f}")

# # Display predictions
print("\nSample Predictions:")
for actual, predicted in zip(y_test[:5], y_pred[:5]):
    print(f"Actual Salary: {actual:.2f}, Predicted Salary: {predicted:.2f}")