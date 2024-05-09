# Social Network Ads Data Analysis

This code performs exploratory data analysis and logistic regression modeling on the Social Network Ads dataset. The dataset contains information about users' age, estimated salary, and whether they purchased a product after seeing an ad.

## Code Overview

The code does the following:

1. Imports necessary libraries like pandas, numpy, matplotlib, seaborn, sklearn

2. Loads the dataset into a pandas dataframe

3. Prints information about the dataframe like shape, columns, statistical summary

4. Splits the data into training and test sets

5. Scales the features using StandardScaler

6. Fits a logistic regression model on the training set

7. Makes predictions on the test set

8. Calculates model evaluation metrics like confusion matrix, classification report

9. Visualizes the model performance on training and test set

## Running the Code

To run this code, you need to have Python installed along with the required libraries mentioned in the imports section.

The dataset 'Social_Network_Ads.csv' also needs to be downloaded into the same folder as this code file.

Then simply run this file in a Python interpreter or IDE like Spyder, Jupyter Notebook, etc.

The visualizations will be displayed inline when running in Jupyter Notebook. For other IDEs, plt.show() will need to be called explicitly to display the plots.

The output will contain various dataframe information, model evaluation metrics, and two visualization plots for training and test set model performance.

## Explanation

The code first loads and explores the dataset. The categorical target variable 'Purchased' and numerical features 'Age', 'EstimatedSalary' are extracted.

The data is split into 75% training and 25% test sets. The features are standardized using StandardScaler to normalize the range.

Logistic regression is fit on the training data. This model is suitable for binary classification problems.

The trained model is used to predict on the unseen test data. Model performance metrics are calculated by comparing predictions to true labels.

The decision boundary is visualized for both training and test set. This gives a sense of how well the model fits each dataset.

Overall the model achieves ~90% accuracy, showing decent performance on this binary classification task. The visualizations also indicate the logistic regression line separating the two classes fairly well.


---

Sure, let's go through each line in detail along with their functions:

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
```
- **Function**: Importing necessary libraries.
- **Explanation**: 
  - `import numpy as np`: Imports NumPy library for numerical computations. Renames it to `np` for convenience in coding.
  - `import matplotlib.pyplot as plt`: Imports the `pyplot` module from Matplotlib for creating plots. Renames it to `plt`.
  - `import pandas as pd`: Imports Pandas library for data manipulation and analysis. Renames it to `pd`.
  - `import seaborn as sns`: Imports Seaborn library for statistical data visualization.

```python
df = pd.read_csv('Social_Network_Ads.csv')
```
- **Function**: Reading a CSV file into a Pandas DataFrame.
- **Explanation**: 
  - `pd.read_csv('Social_Network_Ads.csv')`: Reads a CSV file named 'Social_Network_Ads.csv' into a Pandas DataFrame `df`.

```python
print("----------------Dataframe Info------------------")
print(df.info())
print("\n")
```
- **Function**: Printing DataFrame information.
- **Explanation**:
  - `df.info()`: Provides a concise summary of the DataFrame, including the data types, non-null values, and memory usage.

```python
print("---------------Dataframe Descibe----------------")
print(df.describe())
print("\n")
```
- **Function**: Printing descriptive statistics of the DataFrame.
- **Explanation**:
  - `df.describe()`: Generates descriptive statistics of the numerical columns in the DataFrame, such as count, mean, min, max, etc.

```python
print("---------------First 5 rows of Dataframe----------------")
print(df.head())
print("\n")
```
- **Function**: Printing the first 5 rows of the DataFrame.
- **Explanation**:
  - `df.head()`: Displays the first 5 rows of the DataFrame to get an overview of the data.

```python
print("---------------Train Dataset-------------")
X = df[['Age', 'EstimatedSalary']]
Y = df['Purchased']
```
- **Function**: Creating the feature matrix `X` and target vector `Y`.
- **Explanation**:
  - `df[['Age', 'EstimatedSalary']]`: Selects columns 'Age' and 'EstimatedSalary' from the DataFrame `df` to create the feature matrix `X`.
  - `df['Purchased']`: Selects the column 'Purchased' from the DataFrame `df` to create the target vector `Y`.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(f'Train Dataset Size - X: {X_train.shape}, Y: {Y_train.shape}')
print(f'Test  Dataset Size - X: {X_test.shape}, Y: {Y_test.shape}')
```
- **Function**: Splitting data into training and testing sets, and scaling features.
- **Explanation**:
  - `train_test_split(X, Y, test_size=0.25, random_state=0)`: Splits the data into training and testing sets. `X_train` and `X_test` are feature matrices, and `Y_train` and `Y_test` are target vectors.
  - `StandardScaler()`: Initializes a StandardScaler object to scale the features.
  - `sc_X.fit_transform(X_train)`: Fits the StandardScaler to the training data (`X_train`) and transforms it.
  - `sc_X.transform(X_test)`: Transforms the testing data (`X_test`) using the same scaler.
  - Prints the size of the training and testing datasets.

```python
print("----------------Linner Regression-------------------")
from sklearn.linear_model import LogisticRegression

lm = LogisticRegression(random_state=0, solver='lbfgs')
lm.fit(X_train, Y_train)
predictions = lm.predict(X_test)
plt.figure(figsize=(6, 6))
sns.regplot(x=X_test[:, 1], y=predictions, scatter_kws={'s': 5})
plt.scatter(X_test[:, 1], Y_test, marker='+')
plt.xlabel("User's Estimated Salary")
plt.ylabel('Ads Purchased')
plt.title('Regression Line Tracing')
plt.show()
```
- **Function**: Performing logistic regression and visualizing results.
- **Explanation**:
  - `LogisticRegression(random_state=0, solver='lbfgs')`: Initializes a logistic regression model with a random state for reproducibility and the 'lbfgs' solver.
  - `lm.fit(X_train, Y_train)`: Fits the logistic regression model to the training data.
  - `lm.predict(X_test)`: Makes predictions on the test data.
  - `sns.regplot()`: Plots a regression line between `X_test[:, 1]` (Estimated Salary) and `predictions`.
  - `plt.scatter()`: Plots the actual Purchased values against Estimated Salary.
  - Various `plt` commands set axis labels, title, and display the plot.

```python
print("-------------Confusion Matrix---------------")
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(Y_test, predictions)
print(f'''Confusion matrix :\n
               | Positive Prediction\t| Negative Prediction
---------------+------------------------+----------------------
Positive Class | True Positive (TP) {cm[0, 0]}\t| False Negative (FN) {cm[0, 1]}
---------------+------------------------+----------------------
Negative Class | False Positive (FP) {cm[1, 0]}\t| True Negative (TN) {cm[1, 1]}\n\n''')

cm = classification_report(Y_test, predictions)
print('Classification report : \n', cm)
```
- **Function**: Computing and printing confusion matrix and classification report.
- **Explanation**:
  - `confusion_matrix(Y_test, predictions)`: Computes a confusion matrix to evaluate the accuracy of the classification model.
  - `classification_report(Y_test, predictions)`: Generates a text report showing the main classification metrics.
  - Prints the confusion matrix and classification report.

```python
print("-----------------Visualizing Training set result----------------")
from matplotlib.colors import ListedColormap

X_set, y_set = X_train, Y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.figure(figsize=(9, 7.5))
plt.contourf(X1, X2, lm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.6, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())


for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
- **Function**: Visualizing the training set results.
- **Explanation**:
  - `np.meshgrid()`: Creates a rectangular grid out of two given one-dimensional arrays.
  - `plt.contourf()`: Fills the contour regions using different colors.
  - `plt.scatter()`: Plots points for each class in the training set.
  - Various `plt` commands set axis labels, title, and display the plot.

```python
print("-----------------Visualizing Test set result----------------")
from matplotlib.colors import ListedColormap

X_set, y_set = X_test, Y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01))

plt.figure(figsize=(9, 7.5))
plt.contourf(X1, X2, lm.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.6, cmap=ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                color=ListedColormap(('red', 'green'))(i), label=j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
```
- **Function**: Visualizing the test set results.
- **Explanation**:
  - Similar to the visualization of the training set, but with data from the test set.

  ---


  Here are some potential oral exam questions along with their answers for the given program:

1. **Question**: What does the following code do?

    ```python
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    df = pd.read_csv('Social_Network_Ads.csv')
    ```

    **Answer**: This code imports necessary libraries (NumPy, Matplotlib, Pandas, and Seaborn) and reads a CSV file named 'Social_Network_Ads.csv' into a Pandas DataFrame called `df`.

2. **Question**: Why is it important to print the DataFrame information using `df.info()` and `df.describe()`?

    **Answer**: Printing DataFrame information using `df.info()` provides an overview of the data types, non-null values, and memory usage, while `df.describe()` gives descriptive statistics of numerical columns, such as count, mean, min, max, etc. This helps in understanding the structure and characteristics of the dataset.

3. **Question**: What is the purpose of splitting the data into training and testing sets?

    **Answer**: Splitting the data into training and testing sets using `train_test_split()` helps in evaluating the performance of the machine learning model. The model is trained on the training set and tested on the unseen testing set to assess its generalization ability.

4. **Question**: What is the significance of scaling the features using `StandardScaler()`?

    **Answer**: Scaling the features ensures that each feature contributes equally to the model training process and prevents features with larger scales from dominating the learning algorithm. It helps in improving the convergence rate of the optimization algorithm and the overall performance of the model.

5. **Question**: Explain what logistic regression is and how it is used in this code.

    **Answer**: Logistic regression is a statistical method used for binary classification tasks. It models the probability that a given input belongs to a particular class. In this code, logistic regression is used to predict whether a user will purchase a product based on their age and estimated salary.

6. **Question**: What does the confusion matrix represent in the context of binary classification?

    **Answer**: The confusion matrix is a table that summarizes the performance of a classification model. In a binary classification task, it contains four values: True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN). These values help in understanding the model's predictive accuracy.

7. **Question**: How do you interpret the visualization of the logistic regression results?

    **Answer**: The visualization shows a regression line tracing the relationship between a user's estimated salary and the probability of purchasing the product. The scatter plot displays the actual purchase decisions against estimated salaries. This helps in understanding how well the model predicts the purchase behavior based on salary.

8. **Question**: What is the purpose of visualizing the training and test set results separately?

    **Answer**: Visualizing the training set results helps in understanding how well the model fits the training data, while visualizing the test set results helps in assessing the model's generalization ability to unseen data. It allows us to see if the model is overfitting or underfitting.

9. **Question**: Why is it important to evaluate the performance of the model using both a confusion matrix and a classification report?

    **Answer**: The confusion matrix provides detailed information about the model's predictions, such as true positives, false positives, true negatives, and false negatives, which are essential for understanding the model's accuracy and errors. The classification report provides a summary of key classification metrics like precision, recall, and F1-score, which offer insights into the model's overall performance. Evaluating both helps in getting a comprehensive understanding of the model's effectiveness.

10. **Question**: Can you explain how the code handles data visualization using seaborn and matplotlib?

    **Answer**: The code uses `sns.regplot()` to plot the regression line tracing the relationship between estimated salary and purchase decisions. It also uses `plt.scatter()` to plot the actual purchase decisions against estimated salaries. These visualizations help in understanding the relationship between the features and the target variable. Additionally, it uses other `plt` commands to set axis labels, title, and other visual properties.

These questions cover various aspects of the code, including data handling, model training, evaluation, and visualization. They are designed to assess your understanding of the code's functionality and its application in machine learning tasks.
