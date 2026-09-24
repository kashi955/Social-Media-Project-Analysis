##LIMFADD - Social Media Project Analysis##



**1.Project Overview**



LIMFADD is a machine learning project based on social media account data. The project performs data preprocessing, regression, classification, and dimensionality reduction using Python and Scikit-learn.



The project includes:



 Data cleaning and preprocessing

 Categorical feature encoding

 Linear Regression

 Logistic Regression

 Principal Component Analysis (PCA)

 Data visualization

 Model evaluation



**2. Dataset**



The dataset contains 15,000 records and 11 original features related to social media profiles.



The original features include:



 Followers

 Following

 Following/Followers

 Posts

 Posts/Followers

 Bio

 Profile Picture

 External Link

 Mutual Friends

 Threads

 Labels



**3. Technologies Used**



 Python

 Pandas

 NumPy

 Matplotlib

 Seaborn

 Scikit-learn

 Jupyter Notebook



**4. Data Preprocessing**



The following data preprocessing steps were performed:



1. Cleaned the column names by removing extra spaces and replacing spaces with underscores.

2. Converted numeric-looking values to numeric data types.

3. Checked the dataset for missing values.

4. Replaced invalid values such as `#DIV/0!`, `NaN`, `nan`, and empty values with missing values.

5. Filled missing numeric values using the median.

6. Encoded categorical columns using Label Encoding.

7. Selected relevant features for the machine learning models.

8. Standardized selected features using Standard Scaler before applying PCA.



**5. Machine Learning Models**



The project applies the following machine learning techniques:



**(i)Linear Regression**



Linear Regression was used to predict the `Followers` value.



The ratio-based features `Following/Followers` and `Posts/Followers` were excluded from the model to avoid data leakage.



**Evaluation Metrics:**

R² Score: 0.49695

Mean Squared Error (MSE): 861052527.31539



**(ii)Logistic Regression**



Logistic Regression was used to classify whether the number of followers was high or not based on the median follower value.



The ratio-based features `Following/Followers` and `Posts/Followers` were excluded to avoid data leakage.



**Evaluation:**

 Accuracy: 99.67%

 Confusion Matrix:

 - True Negative: 1501

 - False Positive: 1

 - False Negative: 9

 - True Positive: 1489



**(iii) Principal Component Analysis (PCA)**



PCA was applied after standardizing the selected features using Standard Scaler.



Two principal components were generated for dimensionality reduction and visualization.



**Explained Variance Ratio:**

 Principal Component 1: 34.85%

 Principal Component 2: 18.21%

 Total: 53.06%



**6.Results and Visualization**



The project evaluates the machine learning models using appropriate performance metrics and visualizations.



The analysis includes:



 Linear Regression evaluation using R² Score and Mean Squared Error (MSE).

 Logistic Regression evaluation using accuracy and a confusion matrix.

 PCA visualization using the first two principal components.

 Before-and-after PCA scatter plots for visual comparison.

 PCA explained variance analysis.



The first two principal components explain approximately 53.06% of the total variance in the selected features.



**7.Project Structure**



```text

LIMFADD/

│

├── LIMFADD.csv

├── Limfadd.ipynb

└── README.md



**8.Conclusion**



This project demonstrates a complete machine learning workflow on social media account data, including data preprocessing, feature encoding, regression, classification, dimensionality reduction, and visualization.



The analysis uses Python and Scikit-learn to prepare the data, build machine learning models, evaluate their performance, and visualize the results.

