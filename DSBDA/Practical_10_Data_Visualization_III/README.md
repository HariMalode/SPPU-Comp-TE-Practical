# instructions

1. import the necessary libraries

- commands to install

```
pip install pandas

pip install numpy

pip install seaborn

pip install matplotlib

pip insall scipy
```

2. read the data

3. print the summary of the data

4. print the shape of the data

5. print the first 5 rows of the data

6. print the last 5 rows of the data

7. print the mean of the first column

8. print the histogram of the first column

9. print the histogram of the first column using 5 bins

10. print the columns tilte

11. print the Minimum value from each column

12. print the Maximum value from each column

13. print the Quantile of the database

14. print the correlation between the columns

15. print the frequency of each value in first column

16. print the Density plot for first column

17. print the Heatmap of the correlation of first 4 colums


---

Sure, let's go through the code line by line:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```
These lines import necessary libraries:
- `pandas` as `pd`: This library is used for data manipulation and analysis.
- `numpy` as `np`: It's used for numerical computations.
- `seaborn` as `sns`: It's a Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics.
- `matplotlib.pyplot` as `plt`: It's a plotting library for Python and provides a MATLAB-like interface.

```python
df = pd.read_csv("iris.data")
```
This line reads the data from a CSV file named "iris.data" and stores it in a DataFrame called `df`.

```python
print("--------------Describe the Dataframe----------------------")
print(df.describe())
print("\n")
```
This prints a header and then displays the summary statistics of the DataFrame using the `describe()` function. The `describe()` function calculates count, mean, standard deviation, minimum, 25th percentile, median (50th percentile), 75th percentile, and maximum of numeric columns in the DataFrame.

```python
print("--------------Shape of the Dataframe----------------------")
print(df.shape)
print("\n")
```
This prints the shape of the DataFrame, i.e., the number of rows and columns.

```python
print("--------------First 5 rows of the Dataframe----------------------")
print(df.head())
print("\n")
```
This prints the first 5 rows of the DataFrame using the `head()` function.

```python
print("--------------Last 5 rows of the Dataframe----------------------")
print(df.tail())
print("\n")
```
This prints the last 5 rows of the DataFrame using the `tail()` function.

```python
print("--------------Mean of the First Column----------------------")
print(df["5.1"].mean())
print("\n")
```
This prints the mean of the first column (named "5.1") of the DataFrame using the `mean()` function.

```python
print("--------------Histogram of the Dataframe (using 5 bins)----------------------")
df.hist(bins=5)
plt.show()
print("\n")
```
This creates a histogram of the DataFrame with 5 bins and displays it using `plt.show()`.

```python
print("--------------Histogram of the Dataframe----------------------")
df.hist()
plt.show()
print("\n")
```
This creates a histogram of the DataFrame without specifying the number of bins. It then displays it using `plt.show()`.

```python
print("--------------Columns of the Dataframe----------------------")
print(df.columns)
print("\n")
```
This prints the column names of the DataFrame.

```python
print("--------------Minimum value from Each Column----------------------")
print(df.min())
print("\n")
```
This prints the minimum value from each column of the DataFrame using the `min()` function.

```python
print("--------------Maximum value from Each Column----------------------")
print(df.max())
print("\n")
```
This prints the maximum value from each column of the DataFrame using the `max()` function.

```python
print("--------------Quantile of the Dataframe----------------------")
print(df.quantile([0, 0.25, 0.5, 0.75, 1.0], numeric_only=True))
print("\n")
```
This prints the quantiles (0%, 25%, 50%, 75%, and 100%) of the DataFrame using the `quantile()` function.

```python
print("--------------Correlation of the Dataframe----------------------")
iris_long = pd.melt(df, id_vars='5.1')
ax = sns.boxplot(x="5.1", y="value", hue="variable", data=iris_long)
plt.show()
```
This calculates the correlation between columns of the DataFrame and displays it as a boxplot using Seaborn.

```python
print("--------------Frequecy of each value in the first column----------------------")
print(df['5.1'].value_counts())
print("\n")
```
This prints the frequency of each unique value in the first column using the `value_counts()` function.

```python
print("--------------Density plot for 5.1 column----------------------")
df['5.1'].plot.density(color='green')
plt.title('Density plot for 5.1 column')
plt.xlabel('Value')
plt.ylabel('Density')
plt.show()
print("\n")
```
This creates a density plot for the first column ('5.1') and displays it using `plt.show()`.

```python
print("--------------Heatmap for the Correlation----------------------")
subset_df = df.iloc[:, :4]
plt.figure(figsize=(8, 6))
sns.heatmap(subset_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
```
This creates a heatmap for the correlation matrix of the first four columns of the DataFrame and displays it using Seaborn.


---


Here are some potential questions an examiner might ask you about the program, along with the answers:

1. **What libraries are being imported at the beginning of the program?**
   
   - **Answer:** The program imports four libraries: `pandas`, `numpy`, `seaborn`, and `matplotlib.pyplot`. 
   
2. **What does `pd.read_csv("iris.data")` do?**
   
   - **Answer:** This line reads data from a CSV file named "iris.data" and stores it in a DataFrame called `df`.
   
3. **What information does `df.describe()` provide?**
   
   - **Answer:** `df.describe()` provides summary statistics of the DataFrame, including count, mean, standard deviation, minimum, 25th percentile, median, 75th percentile, and maximum of numeric columns.
   
4. **How would you print the shape of the DataFrame?**
   
   - **Answer:** You would use `print(df.shape)`, which prints the number of rows and columns in the DataFrame.

5. **How do you display the first 5 rows of the DataFrame?**
   
   - **Answer:** You would use `print(df.head())`, which prints the first 5 rows of the DataFrame.

6. **What is the purpose of `df.hist(bins=5)`?**
   
   - **Answer:** It creates a histogram of the DataFrame with 5 bins and displays it using `plt.show()`.

7. **What does `df.min()` do?**
   
   - **Answer:** It prints the minimum value from each column of the DataFrame.

8. **How do you visualize the correlation between columns in the DataFrame?**
   
   - **Answer:** You can visualize the correlation using a heatmap. In this program, it's done with the following lines:
     ```python
     subset_df = df.iloc[:, :4]
     sns.heatmap(subset_df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
     ```
   
9. **What is the difference between `df.hist()` and `df.hist(bins=5)`?**
   
   - **Answer:** `df.hist()` creates histograms of the DataFrame without specifying the number of bins, whereas `df.hist(bins=5)` creates histograms with 5 bins.

10. **How would you calculate the mean of the first column of the DataFrame?**
   
    - **Answer:** You would use `df["5.1"].mean()`.

11. **What is the purpose of `df.quantile([0, 0.25, 0.5, 0.75, 1.0], numeric_only=True)`?**
   
    - **Answer:** It prints the quantiles (0%, 25%, 50%, 75%, and 100%) of the DataFrame.

12. **How do you display the last 5 rows of the DataFrame?**
   
    - **Answer:** You would use `print(df.tail())`, which prints the last 5 rows of the DataFrame.

13. **Explain the line `df['5.1'].plot.density(color='green')`.**
   
    - **Answer:** This line creates a density plot for the column '5.1' and sets the color to green.

14. **What does `df['5.1'].value_counts()` do?**
   
    - **Answer:** It prints the frequency of each unique value in the column '5.1'.