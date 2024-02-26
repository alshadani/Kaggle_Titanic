import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np 

def cleaning(df):
    df_cleaned = df.copy()

    # Calculate average age for males and females
    average_male_age = df_cleaned[df_cleaned['Sex'] == 'male']['Age'].mean()
    average_female_age = df_cleaned[df_cleaned['Sex'] == 'female']['Age'].mean()

    # Fill missing values in 'Age' based on gender
    df_cleaned['Age'] = df_cleaned.apply(
    lambda row: average_male_age if pd.isnull(row['Age']) and row['Sex'] == 'male' else
                  average_female_age if pd.isnull(row['Age']) and row['Sex'] == 'female' else
                  row['Age'],
    axis=1
    )

    # Fill missing values in 'Embarked' with 'C', which is the most common cabin 
    df_cleaned['Embarked'].fillna('C', inplace=True)

    # One-hot encode the 'Embarked' column
    df_cleaned = pd.get_dummies(df_cleaned, columns=['Embarked'], prefix='Embarked')

    df_cleaned['Fare'].fillna(df_cleaned['Fare'].mean(), inplace=True)

    # Encode genders 
    sex_encoded = pd.get_dummies(df_cleaned['Sex'], prefix='Sex', drop_first=True)
    df_cleaned['Sex'] = sex_encoded['Sex_male']

    return df_cleaned

def fearure_engineering(df):
    df = df.copy()
    # Extract titles from the 'Name' column
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
   
    # Combine less common titles into a general category
    df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Capt', 'Countess', 'Lady', 'Sir', 'Jonkheer', 'Don', 'Dona'], 'Other')

    # Encoding of the Cabin column 
    df['Cabin_Present'] = df['Cabin'].notnull().astype(int)

    df['Is_Mr'] = (df['Title'] == 'Mr').astype(int)
    df['Is_Miss'] = (df['Title'] == 'Miss').astype(int)
    df['Is_Mrs'] = (df['Title'] == 'Mrs').astype(int)

    # Feature indicating marital status
    df['Married'] = (df['Title'] == 'Mrs').astype(int)

    # Identify rare titles
    rare_titles = df['Title'].value_counts()[df['Title'].value_counts() < 10].index
    df['Is_Rare_Title'] = df['Title'].isin(rare_titles).astype(int)

    #df = df[df['Fare'] <= 200]

    # Perform Min-Max normalization for 'Fare' column
    min_fare = df['Fare'].min()
    max_fare = df['Fare'].max()
    df['Normalized_Fare'] = 100 * (df['Fare'] - min_fare) / (max_fare - min_fare)

    # Define bins for fare categories
    fare_bins = [0, 25, 50, float('inf')]
    fare_labels = ['0-25', '26-50', '51+']

    # Create a new column 'Fare_Category' based on fare bins
    df['Fare_Category'] = pd.cut(df['Fare'], bins=fare_bins, labels=fare_labels, right=False)

    fare_category_mapping = {'0-25' : 0, '26-50' : 1, '51+' : 2}
    df['Fare_Category'] = df['Fare_Category'].map(fare_category_mapping)

    df['Is_First_fare_category'] = (df['Fare_Category'] == '0-25').astype(int)
    df['Is_Second_fare_category'] = (df['Fare_Category'] == '26-50').astype(int)
    df['Is_Third_fare_category'] = (df['Fare_Category'] == '51+').astype(int)


    # Create 'Family_Size' feature
    df['Family_Size'] = df['Parch'] + df['SibSp']

    # Create 'Is_Alone' feature
    df['Is_Alone'] = (df['Family_Size'] == 0).astype(int)

    # Categorize passengers into different family types
    def categorize_family_size(size):
        if size == 0:
            return 0
        elif size <= 3:
            return 1
        elif size <= 6:
            return 2
        else:
            return 3

    df['Family_Type'] = df['Family_Size'].apply(categorize_family_size)

    df['Is_First_Family_Type'] = (df['Family_Size'] == 0).astype(int)
    df['Is_Second_Family_Type'] = (df['Family_Size'] == 1).astype(int)
    df['Is_Third_Family_Type'] = (df['Family_Size'] == 2).astype(int)
    df['Is_Fourth_Family_Type'] = (df['Family_Size'] == 3).astype(int)

    # Create 'Is_Parent' feature
    df['Is_Parent'] = (df['Parch'] > 0).astype(int)

    # Create 'Is_Child' feature
    df['Is_Child'] = (df['Age'] < 18).astype(int)

    df['Ticket_With_Letters'] = df['Ticket'].str.contains('[a-zA-Z]').astype(int)

    # Pclass encoded 
    df['Is_First_Class'] = (df['Pclass'] == 1).astype(int)
    df['Is_Second_Class'] = (df['Pclass'] == 2).astype(int)
    df['Is_Third_Class'] = (df['Pclass'] == 3).astype(int)


    # Drop Cabin colimn as it has to many NaN values 
    df = df.drop(columns=['Pclass','Family_Type', 'Fare_Category', 'Name', 'Cabin', 'Title', 'Fare', 'Ticket'])

    return df
