# Name: Srinivas Rahul Sapireddy
# Institute: National Institute of Electronics and Information Technology, Calicut, India
# Topic: Churn Analysis
# Software: Python
# IDE: Spyder
# Project Description: The churn rate is helpful for a company to grow its clientele and growth. The churn rate will determine the percentage of customers who will terminate their subscription with the company in a given period. This project is aimed to find the insights from the data to predict the behavior of the customers to stay or leave the company in a given period using machine learning algorithms. 

########## Part 1 - Data Preprocessing ##########

# Importing the libraries
# Load libraries
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Importing the dataset
train_dataset = pd.read_csv('bank_churn.csv')
test_dataset = pd.read_csv('churn_validate.csv')

test_dataset.columns = ['customer_id','credit_score',
                     'country','gender','age','tenure','balance','products_number','credit_card','active_member','estimated_salary','churn']

# ------------------------------------------------------------------------------
########## Part 2 - Analyzing Data ##########
# shape of dataset
print(f"Training Data Dimension: {train_dataset.shape}")
print(f"Test Data Dimension: {test_dataset.shape}")

# Importing the required columns in dataset
X_train = train_dataset.iloc[:, 1:11].values
y_train = train_dataset.iloc[:, 11].values

X_test = test_dataset.iloc[:, 1:11].values
y_test = test_dataset.iloc[:, 11].values

# shape of training dataset
print(f"X_train Data Dimension: {X_train.shape}")
print(f"y_train Data Dimension: {y_train.shape}")
print(f"X_test Data Dimension: {X_test.shape