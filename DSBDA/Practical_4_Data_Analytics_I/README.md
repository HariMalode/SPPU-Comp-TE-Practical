# Boston Housing Data Analysis

## Overview

This code loads the Boston housing dataset, explores it, splits the data into training and test sets, trains a linear regression model to predict housing prices, evaluates the model performance, and provides visualizations.

## Instructions to run

To run this code, you need the following libraries installed:

- numpy
- pandas
- matplotlib
- sklearn

You also need the boston.csv dataset in the same folder as this code.

Simply run this Python file to execute the code from start to finish.

## Code explanation

The code first imports the necessary libraries:

- numpy and pandas for data manipulation
- matplotlib for visualization
- sklearn for machine learning

It then loads the Boston housing dataset into a pandas DataFrame and explores it by printing information, statistics, and a sample of rows.

The data is split into training and test sets using sklearn's train_test_split.

A linear regression model is trained on the training set and predictions are made on the test set.

The model is evaluated by calculating the Mean Squared Error and R^2 Score.

Visualizations are created before and after fitting the model to provide insights.

## Conclusion

The linear regression model achieves a decent performance on this dataset based on the MSE and R^2 values. The visualizations also show the model is able to fit the general trend in the data. This provides a simple example of training and evaluating a machine learning model for a regression problem.


---

Sure, let's go through the code line by line:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

Here, we import necessary libraries: 
- `numpy` for numerical operations
- `pandas` for data manipulation
- `matplotlib.pyplot` for data visualization
- `train_test_split` from `sklearn.model_selection` for splitting data into training and testing sets
- `LinearRegression` from `sklearn.linear_model` for creating a linear regression model
- `mean_squared_error` and `r2_score` from `sklearn.metrics` for evaluation purposes.

```python
boston = pd.read_csv("boston.csv")
```

This line reads a CSV file named "boston.csv" into a pandas DataFrame named `boston`.

```python
print("------------Dataframe Info------------------")
print(boston.info())
print("\n")
```

This prints information about the DataFrame `boston`, including the index dtype and column dtypes, non-null values, and memory usage.

```python
print("----------Dataframe Describe------------")
print(boston.describe())
print("\n")
```

This prints descriptive statistics of the DataFrame `boston`, including count, mean, std (standard deviation), min, 25th percentile, 50th percentile (median), 75th percentile, and max for each numerical column.

```python
print("-----------Dataframe 5 Rows---------------")
print(boston.head())
print("\n")
```

This prints the first 5 rows of the DataFrame `boston`.

```python
print("-----------Dataframe Columns List----------------")
print(boston.columns)
print("\n")
```

This prints the list of columns in the DataFrame `boston`.

```python
X = boston[['RM', 'LSTAT', 'PTRATIO']]
y = boston['MEDV']
```

This creates input features (`X`) and target variable (`y`). Here, `X` contains the columns 'RM', 'LSTAT', and 'PTRATIO', and `y` contains the column 'MEDV'.

```python
print("-------------Splitting data into training and test sets-------------------")
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)
print("\n")
```

This splits the data into training and testing sets using `train_test_split` function from scikit-learn. It prints the shapes of the training and testing sets.

```python
for i, feature in enumerate(X.columns):
    plt.subplot(1, 3, i + 1)
    plt.scatter(X[feature], y, marker='o', s=5)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('MEDV')

plt.tight_layout()
plt.show()
```

This creates scatter plots for each feature against the target variable ('MEDV'). It plots 'RM', 'LSTAT', and 'PTRATIO' on the x-axis and 'MEDV' on the y-axis.

```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

This creates a linear regression model, fits it to the training data (`X_train` and `y_train`), and makes predictions on the test data (`X_test`). 

```python
print("-------------Visualization after fitting model---------------")
# Visualization after fitting the model
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual MEDV')
plt.ylabel('Predicted MEDV')
plt.title('Actual MEDV vs Predicted MEDV')
plt.show()
print("\n")
```

This creates a visualization of the actual vs. predicted values of the target variable ('MEDV').

```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("-----------Evaluation Result----------------")
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
print('\n')
```

This calculates and prints the Mean Squared Error (MSE) and R-squared score (R^2) for the predictions. These are metrics used to evaluate the performance of the regression model.

---


Here are some potential questions an examiner might ask about this program along with their answers:

1. **Question:** What libraries are imported at the beginning of the program?
   - **Answer:** The program imports the following libraries: `numpy`, `pandas`, `matplotlib.pyplot`, `train_test_split` from `sklearn.model_selection`, `LinearRegression` from `sklearn.linear_model`, `mean_squared_error`, and `r2_score` from `sklearn.metrics`.

2. **Question:** What does `pd.read_csv("boston.csv")` do?
   - **Answer:** It reads data from a CSV file named "boston.csv" and stores it in a pandas DataFrame called `boston`.

3. **Question:** What information is printed when `boston.info()` is called?
   - **Answer:** `boston.info()` prints the information about the DataFrame `boston`, including the index dtype and column dtypes, non-null values, and memory usage.

4. **Question:** What does `train_test_split` function do?
   - **Answer:** `train_test_split` function splits the data into training and testing sets. It takes input features (`X`) and target variable (`y`) and returns four sets: `X_train`, `X_test`, `y_train`, `y_test`.

5. **Question:** What are the features (`X`) and target variable (`y`) in this program?
   - **Answer:** The features (`X`) are 'RM', 'LSTAT', and 'PTRATIO', and the target variable (`y`) is 'MEDV'.

6. **Question:** What is the purpose of the scatter plots created using `plt.scatter()`?
   - **Answer:** The scatter plots visualize the relationship between each feature ('RM', 'LSTAT', 'PTRATIO') and the target variable ('MEDV').

7. **Question:** What model is used for regression in this program?
   - **Answer:** A linear regression model is used. It is created using `LinearRegression()` from scikit-learn.

8. **Question:** What evaluation metrics are used to assess the model's performance?
   - **Answer:** Mean Squared Error (MSE) and R-squared score (R^2) are used. They are calculated using `mean_squared_error()` and `r2_score()` from scikit-learn, respectively.

9. **Question:** What is the significance of the visualization created after fitting the model?
   - **Answer:** The visualization compares the actual values of the target variable ('MEDV') with the predicted values. It helps to visually assess the performance of the model.

10. **Question:** How would you interpret the Mean Squared Error (MSE) and R-squared score (R^2) in the context of this model?
   - **Answer:** The Mean Squared Error (MSE) measures the average squared difference between the actual and predicted values of the target variable. A lower MSE indicates better model performance. The R-squared score (R^2) represents the proportion of the variance in the target variable that is explained by the model. It ranges from 0 to 1, with higher values indicating better model fit.
