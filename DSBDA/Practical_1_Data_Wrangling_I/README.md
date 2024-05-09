# Data Wrangling
Data Wrangling, I
Perform the following operations using Python on any open source dataset (e.g., data.csv)
1.   Import all the required Python Libraries.
2.   Locate an open source data from the web (e.g., https://www.kaggle.com). Provide a clear 
description of the data and its source (i.e., URL of the web site).
3.   Load the Dataset into pandas dataframe.
4. Data Preprocessing: check for missing values in the data using pandas isnull(), describe() function to get some initial statistics. Provide variable descriptions. Types of variables etc. Check the dimensions of the data frame.
5. Data Formatting and Data Normalization: Summarize the types of variables by checking the data types (i.e., character, numeric, integer, factor, and logical) of the variables in the data set. If variables are not in the correct data type, apply proper type conversions.
6.   Turn categorical variables into quantitative variables in Python.
In addition to the codes and outputs, explain every operation that you do in the above steps and explain everything that you do to import/read/scrape the data set. 

## Requirements

- Python
- Pandas
- NumPy
- Matplotlib

## Installation

Install the required dependencies:

```bash
pip install pandas numpy matplotlib
```

## Usage

1. Place the dataset `autodata.csv` in the same directory as the script.
2. Run the File `DataWrangling.py`

## Results

The script performs the following tasks:

- Loads the dataset `autodata.csv`.
- Displays basic information about the dataset including data types and non-null values.
- Describes statistical summary of the dataset.
- Shows the first 10 and last 5 rows of the dataset.
- Preprocesses the data by handling missing values.
- Standardizes the data.
- Normalizes the data.
- Converts categorical variables into numerical variables.
- Bins the 'horsepower' column into three bins: Low, Medium, and High.
- Visualizes the binned 'horsepower' data using a histogram and a bar plot.


---

Sure, let's break down the code line by line:

```python
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib import pyplot
```
- Import necessary libraries: 
  - `pandas` for data manipulation and analysis.
  - `numpy` for numerical computing.
  - `matplotlib` for plotting.
  

```python
df = pd.read_csv('autodata.csv')
```
- Reads data from a CSV file named 'autodata.csv' and stores it into a DataFrame called `df`.


```python
print("---------------Information---------------\n")
print(df.info())
print("\n")
```
- Prints information about the DataFrame `df`, such as the data types of each column, non-null counts, and memory usage.


```python
print("---------------Describe the Dataframe---------------")
print(df.describe())
print("\n")
```
- Describes the statistical summary of the DataFrame `df`, including count, mean, standard deviation, minimum, and maximum values of numerical columns.


```python
print("---------------First 10 rows---------------")
print(df.head(10))
print("\n")
```
- Prints the first 10 rows of the DataFrame `df`.


```python
print("---------------Last 5 rows---------------")
print(df.tail())
print("\n")
```
- Prints the last 5 rows of the DataFrame `df`.


```python
print("=================Data Preprocessing======================\n")
```
- Indicates the beginning of the data preprocessing section.


```python
print("---------------Count of Null Values from the Dataset---------------")
print(df.isnull().sum())
print("\n")
```
- Counts and prints the number of null values in each column of the DataFrame `df`.


```python
mean_stroke = df["stroke"].astype("float").mean(axis=0)
print("Average with NULL values : ", mean_stroke)
print("\n")
```
- Calculates the mean value of the "stroke" column and prints it. `.astype("float")` is used to convert the column to float type to avoid errors.


```python
print("---------------Replacing NULL values with mean for 'stroke' column---------------")
df["stroke"] = df["stroke"].replace(np.nan, mean_stroke)
print(df["stroke"])
print("\n")
```
- Replaces null values in the "stroke" column with the calculated mean value.


```python
mean_horsepower = df["horsepower"].astype("float").mean(axis=0)
print("Average : ", mean_horsepower)
print("\n")
```
- Calculates the mean value of the "horsepower" column and prints it.


```python
print("---------------Replacing NULL values with mean for 'horsepower' column---------------")
df["horsepower"] = df["horsepower"].replace(np.nan, mean_horsepower)
print(df["horsepower"])
print("\n")
```
- Replaces null values in the "horsepower" column with the calculated mean value.


```python
print("---------------Filling values in num-of-doors column with mode---------------")
mode = df["num-of-doors"].mode()[0]
print("Mode of the Column : ", mode)
df["num-of-doors"] = df["num-of-doors"].replace(np.nan, mode)
print(df["num-of-doors"])
print("\n")
```
- Fills null values in the "num-of-doors" column with its mode value.


```python
print("---------------Dropping Null value rows---------------")
df.dropna(subset=['horsepower-binned'], axis=0, inplace=True)
df.reset_index(drop=True)
print("number of null rows after : ", df['horsepower-binned'].isnull().sum())
print("\n")
```
- Drops rows with null values in the "horsepower-binned" column.


```python
print("---------------Null value count after Preprocessing---------------")
print(df.isnull().sum())
print("\n")
```
- Prints the count of null values after preprocessing.


```python
print("=================Data Standardization======================\n")
```
- Indicates the beginning of the data standardization section.


```python
print("---------------Standardizing 'city-mpg' column to 'city-L/100km'--------------------")
df["city-L/100km"] = 235/df["city-mpg"]
print(df["city-L/100km"].head())
print("\n")
```
- Standardizes the "city-mpg" column to "city-L/100km" and prints the first five rows of the new column.


```python
print("---------------Standardizing 'highway-mpg' column to 'highway-L/100km'--------------------")
df["highway-L/100km"] = 235/df["highway-mpg"]
print(df["highway-L/100km"].head())
print("\n")
```
- Standardizes the "highway-mpg" column to "highway-L/100km" and prints the first five rows of the new column.


```python
print("=================Data Normalization======================\n")
```
- Indicates the beginning of the data normalization section.


```python
print("---------------Normalizing 'length', 'width', 'height' column--------------------")
df['length']=df['length']/df['length'].max()
df['width']=df['width']/df['width'].max()
df['height']=df['height']/df['height'].max()
print(df[['length','width','height']].head())
print("\n")
```
- Normalizes the 'length', 'width', and 'height' columns and prints the first five rows of these columns.


```python
print("--------------turning categorical values into quantitative (numeric) variables--------------------")
print(df['aspiration'].value_counts())
dummy_var_1=pd.get_dummies(df['aspiration'])
print(dummy_var_1.head())
df=pd.concat([df,dummy_var_1], axis=1)
df.drop('aspiration',axis = 1 , inplace = True)
print(df.head())
```
- Converts categorical values in the 'aspiration' column into dummy variables, concatenates them with the original DataFrame, and drops the 'aspiration' column.

---

Sure, here are some potential questions along with answers:

1. **Question:** What does this Python program do?
   - **Answer:** This Python program performs data preprocessing, standardization, and normalization on a dataset using the Pandas library. It reads data from a CSV file, handles missing values, standardizes columns related to fuel consumption, and normalizes columns related to car dimensions. It also converts categorical variables into numerical variables using one-hot encoding.

2. **Question:** Why is data preprocessing necessary in machine learning and data analysis?
   - **Answer:** Data preprocessing is essential because real-world datasets are often messy and incomplete. Preprocessing helps in cleaning the data, handling missing values, standardizing or normalizing features, and converting categorical variables into a format suitable for machine learning algorithms. It ensures that the data is in a form that allows the model to learn effectively.

3. **Question:** How does the program handle missing values?
   - **Answer:** The program handles missing values by first identifying the columns with missing values using the `isnull().sum()` method. Then, it calculates the mean for numerical columns and mode for categorical columns with missing values. After that, it replaces the missing values in those columns with their respective mean or mode values using the `replace()` method.

4. **Question:** Explain data standardization and normalization.
   - **Answer:** Data standardization is the process of transforming data such that its mean is 0 and standard deviation is 1. In this program, columns related to fuel consumption are standardized by dividing 235 by the values in the 'city-mpg' and 'highway-mpg' columns.
   - Data normalization, on the other hand, scales the data to a range between 0 and 1. In this program, columns related to car dimensions ('length', 'width', 'height') are normalized by dividing each value by the maximum value in its respective column.

5. **Question:** What is one-hot encoding, and why is it used in this program?
   - **Answer:** One-hot encoding is a technique used to convert categorical variables into a numerical format that can be provided to machine learning algorithms. It creates binary columns for each category, where 1 represents the presence of that category and 0 represents its absence. In this program, one-hot encoding is applied to the 'aspiration' column, converting it into two binary columns: 'std' and 'turbo'. This is necessary because machine learning models typically require numerical inputs, and one-hot encoding helps in representing categorical variables appropriately.

   ---
   Data wrangling, also known as data munging, is the process of cleaning, transforming, and preparing raw data into a format suitable for analysis. It involves several steps, including:

1. **Data Collection:** This is the initial step where data is gathered from various sources such as databases, files, APIs, or web scraping.

2. **Data Cleaning:** In this step, data is inspected for errors, missing values, outliers, or inconsistencies. Common tasks include:
   - Handling missing data by imputation, deletion, or interpolation.
   - Removing duplicates.
   - Standardizing formats (e.g., converting data types, correcting typos).
   - Handling outliers.

3. **Data Transformation:** Data transformation involves restructuring or converting data into a usable format. Common tasks include:
   - Encoding categorical variables into numerical format (e.g., one-hot encoding).
   - Scaling or normalizing numerical features to a similar range.
   - Creating new features through feature engineering.
   - Handling text data by tokenization, stemming, or lemmatization.

4. **Data Integration:** If the data comes from multiple sources, it may need to be combined (merged or joined) into a single dataset.

5. **Data Reduction:** Sometimes, datasets may contain unnecessary or redundant information. Data reduction techniques such as dimensionality reduction (e.g., PCA) can be used to simplify the dataset while preserving important information.

6. **Data Formatting:** Data formatting involves organizing data in a structured format suitable for analysis. This may include reshaping data into long or wide format, or preparing it for specific analytical tools or models.

7. **Data Validation:** After the wrangling process, it's important to validate the quality and integrity of the data. This can involve running integrity checks, ensuring data integrity constraints are met, and validating against expected outcomes.

Data wrangling is a crucial step in the data analysis process as it ensures that the data used for analysis is accurate, consistent, and properly formatted. It lays the foundation for meaningful insights and decision-making.
