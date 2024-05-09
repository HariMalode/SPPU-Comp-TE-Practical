# Iris Data Analysis

## Overview

This code loads the iris dataset, splits it into training and test sets, scales the features, trains a Naive Bayes classifier model, makes predictions on the test set, and analyzes the model performance.

Key steps:

- Load iris dataset
- Split data into train/test sets
- Standard scale features
- Train Naive Bayes classifier
- Make predictions on test set
- Map predictions to integers
- Plot regression lines comparing predictions vs actual values
- Print confusion matrix and classification report

## Running the Code

To run this code:

1. Install requirements: pandas, numpy, matplotlib, seaborn, sklearn

2. Run Data_Analytics.py

The output will include dataframe info, statistics, model training and evaluation results.

## Explanation

The dataframe is loaded from the iris CSV file and basic info and statistics are printed.

The data is split into feature data (X) and target data (Y). It is then split into training and test sets and the features are standardized.

A Naive Bayes classifier model is trained on the training data and makes predictions on the test data.

The predictions are mapped to integers and regression lines are plotted to compare the predicted species to the actual species for each feature. This shows how the model performs at predicting species from individual features.

Finally, the confusion matrix and classification report are printed to evaluate the model accuracy, precision, recall etc.

The Naive Bayes model achieves reasonable accuracy on this dataset, but struggles to perfectly separate the different iris species based on the 4 features. The regression plots indicate it does best using the petal features compared to sepal features.

---

Sure, let's break down the code line by line:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```
These lines import the necessary libraries: pandas for data manipulation, numpy for numerical operations, matplotlib.pyplot for plotting, and seaborn for enhanced visualization.

```python
df = pd.read_csv('iris.csv')
```
This line reads the data from the 'iris.csv' file into a pandas DataFrame called `df`.

```python
print("-----------Dataframe Info------------")
print(df.info())
print("\n")
```
These lines print the information about the DataFrame `df`, including the data types and number of non-null values in each column.

```python
print("-----------Dataframe Describe------------")
print(df.describe())
print("\n")
```
These lines print the summary statistics of the DataFrame `df`, including count, mean, min, max, etc., for numerical columns.

```python
print("-----------Dataframe Head------------")  
print(df.head())
print("\n")
```
These lines print the first few rows of the DataFrame `df` to get a glimpse of the data.

```python
print("----------------Data Preprocessing----------------")
X = df.iloc[:,0:4]
Y = df['Species'].values
```
This section prepares the data for machine learning. It separates the features (`X`) and the target variable (`Y`). `X` contains all rows and the first four columns (features), while `Y` contains the 'Species' column which is the target variable.

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
```
These lines import necessary modules for data preprocessing and model building from scikit-learn.

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
```
This line splits the data into training and testing sets using `train_test_split` function. 80% of the data is used for training (`X_train`, `Y_train`), and 20% is used for testing (`X_test`, `Y_test`). `random_state` is set for reproducibility.

```python
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
```
This code standardizes the features by scaling them to have a mean of 0 and a standard deviation of 1 using `StandardScaler`.

```python
print(f'Train Dataset Size - X: {X_train.shape}, Y: {Y_train.shape}')
print(f'Test  Dataset Size - X: {X_test.shape}, Y: {Y_test.shape}')
print("\n")
```
These lines print the sizes of the training and testing datasets.

```python
print("---------------Naive Bayes Classifier----------------------")
```
This line is just a print statement indicating that the next section will deal with a Naive Bayes classifier.

```python
from sklearn.naive_bayes import GaussianNB
```
This line imports the Gaussian Naive Bayes classifier from scikit-learn.

```python
classifier = GaussianNB()
classifier.fit(X_train, Y_train)
predictions = classifier.predict(X_test)
```
These lines create a Gaussian Naive Bayes classifier, fit it to the training data, and make predictions on the test data.

```python
mapper = {'setosa': 0, 'versicolor': 1, 'virginica': 2}
predictions_ = [mapper[i] for i in predictions]
```
This code maps the predicted species labels to integers using a dictionary `mapper`.

```python
fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
_ = fig.suptitle('Regression Line Tracing')
for i in range(4):
    x, y = i // 2, i % 2
    _ = sns.regplot(x=X_test[:, i], y=predictions_, ax=axs[x, y])
    _ = axs[x, y].scatter(X_test[:, i][::-1], Y_test[::-1], marker='+', color="white")
    _ = axs[x, y].set_xlabel(df.columns[i + 1][:-2])
plt.show()
print("\n")
```
This section plots regression lines for each feature against the predicted species labels and actual species labels using Seaborn.

```python
print("------------Confusion Matrix-------------")
```
This line is just a print statement indicating that the next section will deal with a confusion matrix.

```python
from sklearn.metrics import confusion_matrix, classification_report
```
These lines import necessary modules for calculating the confusion matrix and classification report.

```python
cm = confusion_matrix(Y_test, predictions)
print(f'''Confusion matrix :\n
               | Positive Prediction\t| Negative Prediction
---------------+------------------------+----------------------
Positive Class | True Positive (TP) {cm[0, 0]}\t| False Negative (FN) {cm[0, 1]}
---------------+------------------------+----------------------
Negative Class | False Positive (FP) {cm[1, 0]}\t| True Negative (TN) {cm[1, 1]}\n\n''')
```
This code calculates and prints the confusion matrix.

```python
cm = classification_report(Y_test, predictions)
print('Classification report : \n', cm)
```
This code calculates and prints the classification report, which includes precision, recall, F1-score, and support for each class.



---

Sure, here are some potential questions an examiner might ask about this program, along with answers:

**Question 1:** What does this program do?
  
**Answer:** This program performs analysis and classification on the Iris dataset. It first reads the dataset, then performs data preprocessing including splitting the data into training and testing sets, standardizing the features, and finally fits a Naive Bayes classifier to predict the species of Iris flowers.

**Question 2:** How does the program handle data preprocessing?

**Answer:** The program preprocesses the data by splitting it into features (`X`) and the target variable (`Y`). Then it splits the data into training and testing sets using 80-20 ratio. After that, it standardizes the features using `StandardScaler` from scikit-learn.

**Question 3:** What kind of classifier does the program use and why?

**Answer:** The program uses a Gaussian Naive Bayes classifier because it's suitable for classification tasks and assumes that features are independent of each other given the class. It's a simple and efficient algorithm for classification tasks, particularly when dealing with small datasets like the Iris dataset.

**Question 4:** What is the purpose of plotting regression lines in this program?

**Answer:** The program plots regression lines to visualize how well the Naive Bayes classifier predicts the species labels based on each individual feature. It helps in understanding the relationship between features and the predicted species labels.

**Question 5:** Explain the confusion matrix and classification report printed by the program.

**Answer:** The confusion matrix shows the performance of the classifier by comparing the predicted species labels against the actual labels. It includes true positives (correctly predicted positive instances), false positives (incorrectly predicted positive instances), true negatives (correctly predicted negative instances), and false negatives (incorrectly predicted negative instances).

The classification report provides precision, recall, F1-score, and support for each class (species) based on the predictions. It gives insight into the performance of the classifier for each class.
