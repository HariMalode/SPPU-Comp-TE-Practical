
# Academic Performance Dataset Analysis

This Python script analyzes an academic performance dataset. It performs data cleaning, handles missing values, removes outliers, and applies binning to convert data into a normal distribution.

## Instructions

1. **Dependencies**: Ensure you have the following Python libraries installed:

   - pandas
   - numpy
   - matplotlib
   - IPython

2. **Dataset**: Place your academic performance dataset in CSV format in the same directory as this script and name it `Academic-Performance-Dataset.csv`.

3. **Running the Script**:
   - Open the script in a Python environment (e.g., Jupyter Notebook, Python IDE).
   - Run the script. It will perform data analysis and display the results.

## Code Overview

- The script reads the dataset into a pandas DataFrame.
- It performs data cleaning by handling missing values and correcting errors in the dataset.
- Outliers in certain columns are identified and removed.
- Binning is applied to convert the percentage data into grades.

## Note

This script assumes that the input dataset follows a specific format and structure. Ensure that your dataset matches the expected format for accurate analysis.

---

Sure, let's break down the code line by line:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```
- Imports the necessary libraries: `pandas` for data manipulation, `numpy` for numerical operations, `matplotlib.pyplot` for visualization, and `InteractiveShell` from IPython to configure interactive shell settings.

```python
df = pd.read_csv('Academic-Performance-Dataset.csv')
```
- Reads the data from a CSV file named 'Academic-Performance-Dataset.csv' into a pandas DataFrame called `df`.

```python
print("------------------Shape of Dataset------------------")
print(df.shape)
print("\n")
```
- Prints the shape of the DataFrame, which represents the number of rows and columns.

```python
print("-----------------Data Types in Dataset-----------------")
print(df.dtypes)
print("\n")
```
- Prints the data types of each column in the DataFrame.

```python
print("=====================Data Cleanning===================\n")
```
- Indicates the start of the data cleaning process.

```python
print("-------------Handling Missing Values in Dataset-------------")
print(df.isnull().sum())
print("\n")
```
- Prints the sum of missing values in each column of the DataFrame.

```python
print("--------------List of name of columns with missing values--------------")
cols_with_missing = df.columns[df.isnull().any()]
print(cols_with_missing)
print("\n")
```
- Prints the names of columns with missing values.

```python
print("-------------Filling Missing Values with Mean/Mode Imputation-------------")
for col in cols_with_missing:
    col_dt = df[col].dtype
    if col_dt in ['int64', 'float64']:
        outliers = (df[col] < 0) | (df[col] > 100)
        df.loc[outliers, col] = np.nan
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].ffill()
print(df.head())
print("\n")
```
- Handles missing values in the DataFrame:
  - For numerical columns (int64 and float64), it replaces outliers (values less than 0 or greater than 100) with NaN, then fills missing values with the column mean.
  - For other columns, it fills missing values using forward fill (`ffill`).

```python
print("---------------Correction in Percentage & Total Marks Columns---------------")
df['Total Marks']= (df['Phy_marks']+df['Che_marks']+df['EM1_marks']+df['PPS_marks']+df['SME_marks']).astype(int)
df['Percentage']= df['Total Marks']/5
print(df.head())
print("\n")
```
- Calculates the total marks by summing scores from different subjects, and then calculates the percentage based on the total marks. 

```python
print("-------------After Handling Missing Values-------------")
print(df.isnull().sum())
```
- Prints the sum of missing values in each column after handling missing values.

```python
print("=================Handling Outliers===================\n")
```
- Indicates the start of the outlier handling process.

```python
print("-------------Identifying Outliers in Columns-------------")
```
- Prints the message indicating the identification of outliers.

```python
plt.rcParams["figure.figsize"] = (8,5)
df_list = ['Attendence', 'Phy_marks', 'Che_marks', 'EM1_marks', 'PPS_marks', 'SME_marks']
fig, axes = plt.subplots(2, 3)
fig.set_dpi(120)

count  = 0
for r in range(2):
    for c in range(3):
        _ = df[df_list[count]].plot(kind = 'box', ax=axes[r,c])
        count+=1
```
- Sets up a subplot grid for box plots of selected columns to visualize outliers in each column.

```python
print("-------------Removing Outliers from Che_Makrs Column-------------")
q1 = df['Che_marks'].quantile(0.25)
q3 = df['Che_marks'].quantile(0.75)

Lower_limit = q1 - 1.5 * (q3 - q1)
Upper_limit = q3 + 1.5 * (q3 - q1)

print(f'q1 = {q1}, q3 = {q3}, IQR = {q3 -q1}, Lower_limit = {Lower_limit}, Upper_limit = {Upper_limit}')
print(df[(df['Che_marks'] < Lower_limit) | (df['Che_marks'] > Upper_limit)])
print("\n")
```
- Identifies and prints outliers in the 'Che_marks' column based on the interquartile range (IQR) method.

```python
print("=====================Binning (convert into normal distribution)========================\n")
```
- Indicates the start of the binning process.

```python
print("------------Grading According to percentage")
```
- Prints a message indicating the start of grading based on percentage.

```python
def BinningFunction(column, cut_points, labels = None) :
    break_points=[column.min()] + cut_points + [column.max( )]
    print('Grading According to percentage \n<60 = F \n60-70 = B \n70-80 = A\n80-100 = O')
    return pd.cut(column, bins=break_points, labels=labels, include_lowest=True)

cut_points = [60, 70, 80]
labels = ['F', 'B', 'A', 'O']
df['Grade'] = BinningFunction(df['Percentage'], cut_points, labels)
print(df.head(10))
print("\n")
```
- Defines a function `BinningFunction` to bin numerical data into discrete intervals based on cut points.
- Creates cut points and labels for grading percentages into different categories (F, B, A, O).
- Applies the binning function to 'Percentage' column and creates a new column 'Grade' indicating the grade based on percentage.

---
Sure, here are some questions an examiner might ask about this program, along with their answers:

1. **What does this program do?**
   
   Answer: This program performs data cleaning, handling missing values, handling outliers, and binning on a dataset representing academic performance.

2. **What libraries are imported at the beginning of the program, and what are they used for?**

   Answer: The program imports pandas (`pd`) for data manipulation, numpy (`np`) for numerical operations, matplotlib.pyplot (`plt`) for visualization, and InteractiveShell from IPython for configuring interactive shell settings.

3. **How does the program handle missing values?**

   Answer: The program first identifies columns with missing values, then fills missing values using mean imputation for numerical columns and forward fill for categorical columns.

4. **How are outliers identified and handled in this program?**

   Answer: Outliers are identified using box plots for selected columns. Then, for example, in the 'Che_marks' column, outliers are removed based on the interquartile range (IQR) method.

5. **Explain the process of grading based on percentage.**

   Answer: The program defines a function called `BinningFunction` which bins numerical data into discrete intervals based on cut points. Cut points and labels are specified to grade percentages into different categories (F, B, A, O). Then, the 'Percentage' column is binned using this function, and a new column 'Grade' is created indicating the grade based on percentage.

6. **What are the steps involved in data cleaning in this program?**

   Answer: The program first handles missing values by filling them with appropriate methods (mean imputation for numerical columns and forward fill for categorical columns). It then calculates 'Total Marks' and 'Percentage' based on available subject scores, and finally, it corrects any outliers present in the dataset.

7. **Why is it necessary to handle missing values in a dataset?**

   Answer: Handling missing values is crucial as they can adversely affect the analysis and modeling process. Filling missing values ensures that the dataset is complete and accurate, which is essential for making valid conclusions from the data.

8. **What is the significance of binning in data analysis?**

   Answer: Binning is used to convert continuous numerical data into discrete intervals, making it easier to analyze and interpret. It helps in simplifying complex distributions, identifying patterns, and facilitating comparisons between different groups or categories. In this program, binning is used to grade student percentages into categories for easier interpretation.