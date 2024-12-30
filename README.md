
# Salary Prediction with Linear Regression

## **Project Overview**
This project implements a simple linear regression model from scratch in Python to predict employee salaries based on their education, job title, gender, and location. The implementation avoids using machine learning libraries like `scikit-learn` for training, focusing on understanding the inner workings of linear regression.

---

## **Dataset**
The dataset includes the following features:
- **Education:** Education levels like High School, Bachelor, Master, and PhD.
- **Job Title:** Positions like Analyst, Engineer, Manager, and Director.
- **Gender:** Male or Female.
- **Location:** Rural, Suburban, or Urban.
- **Salary:** The target variable representing the employee's salary.

### **Preprocessing**
Categorical variables are encoded into numerical values. For example:
- Education levels are mapped as: `{'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}`.
- Job titles, gender, and location are similarly encoded.

Encoded features are added to the dataset, and the original columns are dropped.

---

## **Code Implementation**
### **1. Feature and Target Separation**
The features (`X`) include all encoded columns except the target (`Salary`), which is stored as the target variable (`y`).

```python
X = data.drop(columns=["Education", "Gender", "Location", "Job_Title", "Salary"])
y = data["Salary"].values
```

### **2. Train-Test Split**
The dataset is split into training and testing sets using a custom function with a default 80-20 split.

```python
def split_data(X, y, split_ratio=0.8):
    split_index = int(len(data) * split_ratio)
    X_train = X[:split_index]
    X_test = X[split_index:]
    y_train = y[:split_index]
    y_test = y[split_index:]
    return X_train, X_test, y_train, y_test
```

---

### **3. Linear Regression Implementation**
A custom `LinearRegression` class is implemented with the following methods:
1. **`fit(X, y)`**: Trains the model using gradient descent.
2. **`predict(X)`**: Predicts target values for new data.
3. **`evaluate(X, y)`**: Calculates Mean Squared Error (MSE) to evaluate model performance.

#### **Key Features**
- Data standardization during training and prediction.
- Gradient clipping to stabilize the training process.

```python
class LinearRegression:
    def fit(self, X, y):
        # Standardize data and implement gradient descent
        pass

    def predict(self, X):
        # Make predictions and unstandardize results
        pass

    def evaluate(self, X, y):
        # Calculate Mean Squared Error
        pass
```

---

### **4. Model Training and Evaluation**
The model is trained on the training set and evaluated on the testing set.

```python
# Train the model
model = LinearRegression(learning_rate=0.01, epoch=1000)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = model.evaluate(X_test, y_test)

print(f"Mean Squared Error: {mse:.2f}")
```

Sample predictions are displayed for review.

---

## **How to Run the Code**
1. Clone this repository and ensure the dataset (`salary_prediction_data.csv`) is present in the working directory.
2. Run the script using Python:
   ```bash
   python main.py
   ```
3. Review the output for MSE and sample predictions.

---

## **Project Highlights**
- **Custom Linear Regression**: Built from scratch, avoiding external ML libraries.
- **Data Preprocessing**: Encodes categorical variables and standardizes data.
- **Gradient Descent**: Implements optimization with gradient clipping for stable learning.
