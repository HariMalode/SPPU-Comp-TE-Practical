# Data Visualization II

## Overview

This Python script uses Pandas and Seaborn to load and explore the Titanic dataset. It generates some basic visualizations to understand the data including:

- Boxplot of Age vs Gender
- Boxplot of Age vs Gender, colored by Survived

## Installation

This script requires Pandas and Seaborn. Install via pip:

pip install pandas seaborn

## Code Explanation

The script first loads and prints the Titanic dataset using Seaborn.

It then generates a boxplot showing Age on the y-axis and Gender on the x-axis. This allows us to see the distribution of Age by Gender.

Next, it makes the same boxplot but colors the boxes based on the Survived column. This allows us to see the Age distribution of those who survived vs those who did not, broken down by Gender.

Some observations:

- Women have a wider spread of Ages than Men
- The median Age for Men is higher than for Women
- Surviving Women skewed younger compared to those who did not survive
- Surviving and non-surviving Men have a more similar Age distribution

---

Sure! Let's go through the code line by line:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

1. `import numpy as np`: This line imports the numpy library and gives it the alias `np`. Numpy is a library for numerical computing in Python.

2. `import pandas as pd`: This line imports the pandas library and gives it the alias `pd`. Pandas is a library for data manipulation and analysis.

3. `import matplotlib.pyplot as plt`: This line imports the `pyplot` module from the `matplotlib` library and gives it the alias `plt`. Matplotlib is a plotting library in Python.

4. `import seaborn as sns`: This line imports the seaborn library and gives it the alias `sns`. Seaborn is a statistical data visualization library based on matplotlib.

```python
ds = sns.load_dataset('titanic')
```

5. `ds = sns.load_dataset('titanic')`: This line loads the Titanic dataset using seaborn's `load_dataset` function and assigns it to the variable `ds`. The Titanic dataset contains information about passengers on the Titanic.

```python
print("-----------Dataset first 5 rows-------------")
print(ds.head())
print("\n")
```

6. `print("-----------Dataset first 5 rows-------------")`: This line prints a header indicating that the following output will be the first 5 rows of the dataset.

7. `print(ds.head())`: This line prints the first 5 rows of the dataset `ds`. The `head()` function in pandas is used to display the first few rows of a DataFrame.

8. `print("\n")`: This line prints a newline character to add space between outputs.

```python
print("-------Boxplot Gender vs Age---------------")
sns.boxplot(x='sex', y='age', data=ds)
plt.show()
print("\n")
```

9. `print("-------Boxplot Gender vs Age---------------")`: This line prints a header indicating that the following plot will be a boxplot showing the relationship between gender and age.

10. `sns.boxplot(x='sex', y='age', data=ds)`: This line creates a boxplot using seaborn's `boxplot` function. It plots age (y-axis) against gender (x-axis) using the data from the dataset `ds`.

11. `plt.show()`: This line displays the plot. `plt.show()` is a function from matplotlib that shows all the figures created until this point.

12. `print("\n")`: This line prints a newline character to add space between outputs.

```python
print("-------------Survived Passengers---------------------")
sns.boxplot(x='sex', y='age', data=ds, hue='survived')
plt.show()
print("\n")
```

13. `print("-------------Survived Passengers---------------------")`: This line prints a header indicating that the following plot will show the distribution of survived passengers.

14. `sns.boxplot(x='sex', y='age', data=ds, hue='survived')`: This line creates a boxplot showing the relationship between gender, age, and survival status. The parameter `hue='survived'` colors the boxes based on the survival status of the passengers.

15. `plt.show()`: This line displays the plot.

16. `print("\n")`: This line prints a newline character to add space between outputs.

These lines of code essentially import necessary libraries, load the Titanic dataset, print some information about the dataset, and create boxplots to visualize relationships within the data.

---

Sure, here are some potential questions an examiner might ask about the program along with their answers:

1. **Question:** What libraries are imported in this program and what are their purposes?
   
   **Answer:** This program imports four libraries: numpy (`np`), pandas (`pd`), matplotlib.pyplot (`plt`), and seaborn (`sns`). Numpy and pandas are used for data manipulation and analysis, matplotlib.pyplot is used for plotting, and seaborn is used for statistical data visualization.

2. **Question:** What does the line `ds = sns.load_dataset('titanic')` do?
   
   **Answer:** This line loads the Titanic dataset using seaborn's `load_dataset` function and assigns it to the variable `ds`.

3. **Question:** What does `ds.head()` do and why is it used here?
   
   **Answer:** `ds.head()` prints the first 5 rows of the dataset `ds`. It's used here to give a quick overview of the data and its structure.

4. **Question:** Explain the purpose of the code `sns.boxplot(x='sex', y='age', data=ds)` followed by `plt.show()`.

   **Answer:** This code creates a boxplot showing the relationship between gender and age using the data from the Titanic dataset. `sns.boxplot()` is used to create the boxplot, and `plt.show()` is used to display the plot.

5. **Question:** What does the parameter `hue='survived'` in `sns.boxplot()` do?
   
   **Answer:** The `hue='survived'` parameter colors the boxes in the boxplot based on the survival status of the passengers, allowing for visualization of the distribution of survived and non-survived passengers within each gender group.

6. **Question:** What is the purpose of the blank lines (`print("\n")`) in the code?
   
   **Answer:** The blank lines add space between different sections of output, making it easier to read and understand.

7. **Question:** If you wanted to visualize another aspect of the Titanic dataset, how would you modify this code?
   
   **Answer:** To visualize another aspect, you could change the variables passed to `sns.boxplot()`. For example, if you wanted to visualize the distribution of fares among different passenger classes, you could change `x='sex'` to `x='class'` and `y='fare'`.
