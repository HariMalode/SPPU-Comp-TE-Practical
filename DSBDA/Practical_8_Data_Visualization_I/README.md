# DataAnalytics.py

This Python script performs some basic data analytics and visualization on the Titanic dataset using Pandas, Seaborn, and Matplotlib.

## Overview

The script loads the Titanic dataset from Seaborn and prints out the first few rows to display the structure. It then sets the Seaborn style to "whitegrid" for nicer plot styling.

Two visualizations are created:

1. A bar plot showing survival rate by gender and passenger class. This uses the `sns.catplot()` function.

2. A histogram of ticket prices using `sns.histplot()`.

## Running the Code

To run this script, you need Python 3 along with Pandas, Seaborn, and Matplotlib installed.

Simply execute:

python DataAnalytics.py

This will load the data, create the plots, and display them.

## Explanation

The bar plot clearly shows that females had a much higher survival rate than males, especially for the First and Second class passengers. The Third class females still had a noticeable higher survival rate than the males.

The ticket price histogram shows the distribution of fares paid. It's right skewed, meaning most passengers paid low fares, with a long tail to the right of higher fares. The peak frequency is around 10.

Together, these visualizations help uncover patterns in the Titanic dataset regarding survival rates and ticket prices. The code provides a simple starting point for further data analysis and visualization.


---

Sure, let's break down the code line by line:

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

1. `import seaborn as sns`: This imports the Seaborn library, which is used for statistical data visualization.
2. `import matplotlib.pyplot as plt`: This imports the `pyplot` module from the Matplotlib library, which is used for creating visualizations.
3. `import pandas as pd`: This imports the Pandas library, which is used for data manipulation and analysis.

```python
titanic = sns.load_dataset('titanic')
```

4. `titanic = sns.load_dataset('titanic')`: This loads the Titanic dataset from the Seaborn library and assigns it to the variable `titanic`. The Titanic dataset contains information about passengers on the Titanic, including their demographics and survival status.

```python
print("-------------Dataset first 5 rows-------------")
print(titanic.head())
print("\n")
```

5. `print("-------------Dataset first 5 rows-------------")`: This prints a descriptive message indicating that the following output displays the first 5 rows of the dataset.
6. `print(titanic.head())`: This prints the first 5 rows of the `titanic` DataFrame.
7. `print("\n")`: This prints a newline character for better readability.

```python
print("Setting style to whitegrid")
sns.set_style("whitegrid")
```

8. `print("Setting style to whitegrid")`: This prints a descriptive message indicating that the plotting style is being set.
9. `sns.set_style("whitegrid")`: This sets the plotting style to "whitegrid" using Seaborn. This style provides a white background with grid lines.

```python
print("--------------Creating bar plot of survival rate by gender and class------------------")
sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)
plt.title('Survival Rate by Gender and Class')
plt.show()
print("\n")
```

10. `print("--------------Creating bar plot of survival rate by gender and class------------------")`: This prints a descriptive message indicating the type of plot that is being created.
11. `sns.catplot(x="sex", y="survived", hue="class", kind="bar", data=titanic)`: This creates a categorical plot (bar plot) using Seaborn's `catplot` function. It shows the survival rate (`survived`) based on gender (`sex`) and class (`class`) of passengers from the Titanic dataset.
12. `plt.title('Survival Rate by Gender and Class')`: This sets the title of the plot.
13. `plt.show()`: This displays the plot.
14. `print("\n")`: This prints a newline character for better readability.

```python
print("--------------Plotting histogram of ticket prices-------------------------")
plt.figure(figsize=(10, 6))
sns.histplot(data=titanic, x='fare', bins=30, kde=True)
plt.title('Distribution of Ticket Prices')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
```

15. `print("--------------Plotting histogram of ticket prices-------------------------")`: This prints a descriptive message indicating the type of plot that is being created.
16. `plt.figure(figsize=(10, 6))`: This creates a new figure with a specified size.
17. `sns.histplot(data=titanic, x='fare', bins=30, kde=True)`: This creates a histogram plot using Seaborn's `histplot` function. It shows the distribution of ticket prices (`fare`) from the Titanic dataset. `bins=30` specifies the number of bins, and `kde=True` adds a kernel density estimate to the plot.
18. `plt.title('Distribution of Ticket Prices')`: This sets the title of the plot.
19. `plt.xlabel('Fare')`: This sets the label for the x-axis.
20. `plt.ylabel('Frequency')`: This sets the label for the y-axis.
21. `plt.show()`: This displays the plot.

Each line in the code plays a specific role in loading the data, setting up the visualizations, and displaying the plots. Let me know if you need further clarification on any part!

---

Certainly! Here's a set of questions an examiner might ask you about the code, along with their answers:

1. **Question:** What libraries are being imported at the beginning of the code?
   - **Answer:** The code imports three libraries: Seaborn (`import seaborn as sns`), Matplotlib (`import matplotlib.pyplot as plt`), and Pandas (`import pandas as pd`).

2. **Question:** How is the Titanic dataset loaded into the program?
   - **Answer:** The Titanic dataset is loaded using the `sns.load_dataset('titanic')` function from the Seaborn library. It is stored in a variable named `titanic`.

3. **Question:** What is the purpose of the `sns.set_style("whitegrid")` line?
   - **Answer:** This line sets the plotting style to "whitegrid", which provides a white background with grid lines for the subsequent plots.

4. **Question:** Describe the first plot created in the code.
   - **Answer:** The first plot is a bar plot showing the survival rate by gender and class of passengers from the Titanic dataset. It uses the `sns.catplot()` function to create the plot and displays it with the title "Survival Rate by Gender and Class".

5. **Question:** How is the second plot different from the first one?
   - **Answer:** The second plot is a histogram showing the distribution of ticket prices from the Titanic dataset. It uses the `sns.histplot()` function to create the plot and displays it with the title "Distribution of Ticket Prices".

6. **Question:** What is the purpose of the `plt.figure(figsize=(10, 6))` line?
   - **Answer:** This line creates a new figure with a specific size of 10 inches in width and 6 inches in height. It ensures that the histogram plot will be displayed in a larger size.

7. **Question:** Explain the significance of the `kind="bar"` parameter in the `sns.catplot()` function.
   - **Answer:** The `kind="bar"` parameter specifies the type of plot to be created, in this case, a bar plot. It tells Seaborn to create a bar plot for the data provided.

8. **Question:** How are the x-axis and y-axis labels set for the histogram plot?
   - **Answer:** The x-axis label is set using `plt.xlabel('Fare')`, and the y-axis label is set using `plt.ylabel('Frequency')`.

9. **Question:** Why is the `kde=True` parameter used in the `sns.histplot()` function?
   - **Answer:** The `kde=True` parameter adds a kernel density estimate to the histogram plot, providing additional information about the distribution of the data.

10. **Question:** What is the purpose of the `plt.show()` function?
   - **Answer:** The `plt.show()` function is used to display the plot on the screen. It should be called after all the plotting commands to show the final result.

These questions cover various aspects of the code, from libraries and data loading to plot creation and customization. Understanding these concepts will give you a good grasp of the code and its functionality.
