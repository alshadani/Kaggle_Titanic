import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

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
    embarked_mapping = {'C': 0, 'Q': 1, 'S': 2}
    df_cleaned['Embarked'] = df_cleaned['Embarked'].map(embarked_mapping).astype(int)

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

    title_mapping = {'Mr': 0, 'Miss': 1, 'Mrs': 2}
    df['Title_Encoded'] = df['Title'].map(title_mapping).fillna(3).astype(int)

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

    # Create 'Is_Parent' feature
    df['Is_Parent'] = (df['Parch'] > 0).astype(int)

    # Create 'Is_Child' feature
    df['Is_Child'] = (df['Age'] < 18).astype(int)

    df['Ticket_With_Letters'] = df['Ticket'].str.contains('[a-zA-Z]').astype(int)

    # Create the 'Pclass_Embarked' feature by combining 'Pclass' and 'Embarked'
    df['Pclass_Embarked'] = df['Pclass'].astype(str) + '_' + df['Embarked'].astype(str)

    # Use LabelEncoder to encode 'Pclass_Embarked'
    label_encoder = LabelEncoder()
    df['Pclass_Embarked_Encoded'] = label_encoder.fit_transform(df['Pclass_Embarked'])

    # Create a new feature by combining 'Ticket_With_Letters' and 'Pclass'
    df['Ticket_Pclass_Combined'] = df['Ticket_With_Letters'].astype(str) + '_' + df['Pclass'].astype(str)

    # Use LabelEncoder to encode the combined feature
    df['Ticket_Pclass_Combined_Encoded'] = label_encoder.fit_transform(df['Ticket_Pclass_Combined'])

    # Drop Cabin colimn as it has to many NaN values 
    df = df.drop(columns=['Name', 'Cabin', 'Title', 'Fare', 'Ticket', 'Ticket_Pclass_Combined', 'Pclass_Embarked', 'Ticket_Pclass_Combined'])
    return df
