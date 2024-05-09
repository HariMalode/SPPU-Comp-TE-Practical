# 3.1 Wage Dataset Analysis

This Python script analyzes a wage dataset, providing various statistical insights into the data. The dataset contains information about wages and demographics, and the script explores different aspects of the dataset to gain insights into wage distribution, age demographics, and educational background.

## Instructions

1. **Dependencies**: Ensure you have the pandas library installed. You can install it using `pip install pandas`.

2. **Dataset**: Place your wage dataset in CSV format in the same directory as this script and name it `wages.csv`.

3. **Running the Script**:
   - Open the script in a Python environment (e.g., Jupyter Notebook, Python IDE).
   - Run the script. It will perform analysis on the dataset and display the results.

## Analysis Performed

The script performs the following analyses:

- **Descriptive Statistics**: Provides descriptive statistics for the entire dataset, including count, mean, standard deviation, minimum, and maximum values.
- **Shape and Size**: Prints the shape and size of the dataset.
- **Minimum and Maximum Values**: Prints the minimum and maximum values for each column in the dataset.
- **Mean, Median, and Mode of Age Column**: Calculates and prints the mean, median, and mode of the age column.
- **Standard Deviation of Age Column**: Calculates and prints the standard deviation of the age column.
- **Grouped Statistics by Education Level**: Groups the age column by education level and provides descriptive statistics for each group.

## Note

This script assumes that the input dataset (`wages.csv`) follows a specific format and structure as described above. Ensure that your dataset matches the expected format for accurate analysis.

# 3.2 Iris Dataset Analysis

This Python script analyzes the Iris dataset, which contains measurements of various iris flowers. The dataset is a popular dataset in the field of machine learning and is often used for classification tasks.

## Instructions

1. **Dependencies**: Ensure you have the pandas, numpy, and matplotlib libraries installed. You can install them using `pip install pandas numpy matplotlib`.

2. **Dataset**: Place your Iris dataset in CSV format in the same directory as this script and name it `iris.csv`.

3. **Running the Script**:
   - Open the script in a Python environment (e.g., Jupyter Notebook, Python IDE).
   - Run the script. It will perform analysis on the dataset and display the results.

## Analysis Performed

The script performs the following analyses:

- **Dataframe Info**: Provides information about the dataset, including column names, data types, and memory usage.
- **Dataframe Shape**: Prints the shape (number of rows and columns) of the dataset.
- **Dataframe Describe**: Provides descriptive statistics for the numerical columns in the dataset, including count, mean, standard deviation, minimum, and maximum values.
- **Basic Statistics by Species**:
  - Provides basic statistics for each species of iris flower (setosa, versicolor, virginica), including count, mean, standard deviation, minimum, and maximum values for each numerical feature.

## Note

This script assumes that the input dataset (`iris.csv`) follows a specific format and structure as described above. Ensure that your dataset matches the expected format for accurate analysis.


---

Sure, here's a breakdown of each line of the code:

```python
import pandas as pd
```
This line imports the Pandas library and gives it the alias `pd`, making it easier to reference in the code.

```python
df = pd.read_csv('wages.csv')
```
This line reads a CSV file named `'wages.csv'` into a DataFrame called `df`. `read_csv()` is a function from the Pandas library used to read CSV files and create a DataFrame.

```python
print("-------------Dataframe Described-------------")
print(df.describe())
print("\n")
```
These lines print out the summary statistics of the DataFrame `df` using the `describe()` method. This method generates various summary statistics, such as count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for numeric columns.

```python
print("-------------Dataframe Shape-----------------")
print(df.shape)
print("\n")
```
These lines print out the shape of the DataFrame `df`, which is a tuple representing the number of rows and columns in the DataFrame.

```python
print("-------------Dataframe Size-----------------")
print(df.size)
print("\n")
```
These lines print out the total number of elements in the DataFrame `df`, which is the product of the number of rows and columns.

```python
print("-------------Dataframe Min Value-----------------")
print(df.min())
print("\n")
```
These lines print out the minimum value for each column in the DataFrame `df`.

```python
print("-------------Dataframe Max Value-----------------")
print(df.max())
print("\n")
```
These lines print out the maximum value for each column in the DataFrame `df`.

```python
print("-------------Age Column Mean-----------------")
print(df['age'].mean())
print("\n")
```
This line prints out the mean of the 'age' column in the DataFrame `df`.

```python
print("-------------Age Column Median-----------------")
print(df['age'].median())
print("\n")
```
This line prints out the median of the 'age' column in the DataFrame `df`.

```python
print("-------------Age Column Mode-----------------")
print(df['age'].mode())
print("\n")
```
This line prints out the mode of the 'age' column in the DataFrame `df`.

```python
print("-------------Age Column Standard Deviation-----------------")
print(round(df['age'].std(),4))
print("\n")
```
This line prints out the standard deviation of the 'age' column in the DataFrame `df`, rounded to four decimal places.

```python
print("-------------Age Column Descibe-----------------")
print(round(df['age'].describe(),3))
print("\n")
```
This line prints out the summary statistics of the 'age' column in the DataFrame `df`, rounded to three decimal places.

```python
print("-------------Age Column Grouped By Education (Describe)-----------------")
print(df['age'].groupby(df['ed']).describe())
print("\n")
```

This line groups the 'age' column by the 'ed' column (education level) and then prints out the summary statistics for each group.

---


Here's the explanation of each line of the code:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```
These lines import the Pandas library as `pd`, the NumPy library as `np`, and the `pyplot` module from the Matplotlib library as `plt`. Pandas is used for data manipulation and analysis, NumPy for numerical computing, and Matplotlib for data visualization.

```python
df = pd.read_csv('iris.csv')
```
This line reads the CSV file named `'iris.csv'` into a Pandas DataFrame called `df`. The `read_csv()` function is used to read CSV files in Pandas.

```python
print("------------------DataframeInfo-----------------")
print(df.info())
print("\n")
```
These lines print out concise summary information about the DataFrame `df`, including the number of non-null entries in each column and the data types of each column. The `info()` method provides a concise summary of the DataFrame.

```python
print("-----------------Dataframe Shape-----------------")
print(df.shape)
print("\n")
```
These lines print out the shape of the DataFrame `df`, which is a tuple representing the number of rows and columns.

```python
print("--------------Dataframe Describe---------------------")
print(df.describe())
print("\n")
```
These lines print out the summary statistics of the DataFrame `df`. The `describe()` method generates various summary statistics, such as count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum for numeric columns.

```python
print("-----------------Basic Statistics (Iris-setosa)-----------------")
df_setosa = df["Species"] == "Iris-setosa"
print(df[df_setosa].describe())
print("\n")
```
These lines filter the DataFrame `df` to only include rows where the "Species" column is equal to "Iris-setosa", then print out the summary statistics of these filtered rows. It describes basic statistics for the Iris-setosa species.

```python
print("-----------------Basic Statistics (Iris-versicolor)-----------------")
df_versicolor = df["Species"] == "Iris-versicolor"
print(df[df_versicolor].describe())
print("\n")
```
These lines filter the DataFrame `df` to only include rows where the "Species" column is equal to "Iris-versicolor", then print out the summary statistics of these filtered rows. It describes basic statistics for the Iris-versicolor species.

```python
print("-----------------Basic Statistics (Iris-virginica)-----------------")
df_virginica = df["Species"] == "Iris-virginica"
print(df[df_virginica].describe())
print("\n")
```
These lines filter the DataFrame `df` to only include rows where the "Species" column is equal to "Iris-virginica", then print out the summary statistics of these filtered rows. It describes basic statistics for the Iris-virginica species.

----

Sure, here are some potential questions the examiner might ask about the code along with their answers:

**Question 1:** What libraries are being imported at the beginning of the code?
  
**Answer:** The code imports three libraries: Pandas (as `pd`), NumPy (as `np`), and Matplotlib's pyplot module (as `plt`).

**Question 2:** What does `pd.read_csv('iris.csv')` do?

**Answer:** It reads the CSV file named 'iris.csv' into a Pandas DataFrame called `df`.

**Question 3:** What does `df.info()` do?

**Answer:** It provides concise information about the DataFrame, including the number of non-null entries in each column and the data types of each column.

**Question 4:** What is the shape of the DataFrame `df`?

**Answer:** The shape of the DataFrame `df` is `(150, 5)`, meaning it has 150 rows and 5 columns.

**Question 5:** What does `df.describe()` do?

**Answer:** It generates summary statistics of the DataFrame, including count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum for numeric columns.

**Question 6:** How are the basic statistics for each species calculated?

**Answer:** They are calculated by filtering the DataFrame `df` for each species using boolean indexing (`df["Species"] == "Iris-setosa"`, `df["Species"] == "Iris-versicolor"`, `df["Species"] == "Iris-virginica"`) and then applying the `describe()` method to each filtered DataFrame.

**Question 7:** Why do we use boolean indexing (`df["Species"] == "Iris-setosa"`, etc.)?

**Answer:** Boolean indexing is used to filter the DataFrame based on a condition, in this case, filtering rows where the "Species" column matches the specified species.

**Question 8:** What information does the program provide for each species?

**Answer:** For each species, the program provides basic statistics such as count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum for numeric columns.

**Question 9:** Why do we use `print("\n")` after each output?

**Answer:** `print("\n")` is used to print a blank line after each output for better readability and separation of results.

**Question 10:** What visualization techniques could be used to further analyze this data?

**Answer:** Histograms, box plots, or scatter plots could be used to visualize the distribution of features across different species. Matplotlib, which was imported as `plt`, could be used for this purpose.