# Process and Visualize data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from scipy import stats, interp
from sqlalchemy import create_engine
import seaborn as sns
import datetime

# Preprocess data
from sklearn.preprocessing import LabelEncoder, label_binarize, OneHotEncoder, StandardScaler

# Feature selection
from sklearn.decomposition import PCA

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, make_scorer, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

# Read the full contents of the dataframes
pd.set_option('display.max_colwidth', None)
pd.options.display.max_columns = None


# Define a function to shift columns to the right for rows with null species
def shift_row(row, avg_sepal_length):
    """
    Shifts columns to the right for rows with null sepal_length, replacing
    the null sepal_length with the mean sepal_length.

    Parameters:
    - row: pd.Series
        The row of the DataFrame.
    - avg_sepal_length: Double
        The average sepal length of the flower species

    Returns:
    - pd.Series
        The shifted row.
    """
    if pd.isnull(row['species']):
        return pd.Series([row['petal_width'], row['petal_length'],
                          row['sepal_width'], row['sepal_length'],
                          avg_sepal_length],
                         index=['species', 'petal_width', 'petal_length',
                                'sepal_width', 'sepal_length'])
    return row


def convert_columns_to_numeric(df, columns):
    """
    Convert specified columns of a DataFrame to numeric, coercing errors to NaN.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the columns to convert.
    columns (list of str): The list of column names to convert to numeric.

    Returns:
    pandas.DataFrame: The DataFrame with specified columns converted to numeric.
    """
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df


def evaluate_classifiers(X_train, X_test, y_train, y_test):
    """
    Evaluate multiple classifiers on the given training and test datasets.

    Parameters:
    - X_train: pd.DataFrame
        Training features.
    - X_test: pd.DataFrame
        Test features.
    - y_train: pd.Series
        Training labels.
    - y_test: pd.Series
        Test labels.

    Returns:
    - output: pd.DataFrame
        DataFrame containing the performance metrics of each classifier.
    """

    # Preprocessing
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X_train.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)])

    # Define classifiers
    classifiers = {
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Support Vector Machine': LinearSVC(max_iter=10000),
        'Decision Tree': DecisionTreeClassifier(),
        'Bagging Decision Tree (Ensemble)': BaggingClassifier(DecisionTreeClassifier(), max_samples=0.5,
                                                              max_features=1.0, n_estimators=10),
        'Boosting Decision Tree (Ensemble)': AdaBoostClassifier(
            DecisionTreeClassifier(min_samples_split=10, max_depth=4),
            n_estimators=10, learning_rate=0.6),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=30, max_depth=8),
        'Voting Classifier (Ensemble)': VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=8)),
            ('svm', LinearSVC(max_iter=10000))
        ], voting='hard')
    }

    scoring = {
        'accuracy': 'accuracy',
        'precision': make_scorer(precision_score, average='weighted'),
        'recall': make_scorer(recall_score, average='weighted'),
        'f1': make_scorer(f1_score, average='weighted'),
    }

    results = []
    predictions = {}

    for label, clf in classifiers.items():
        start = datetime.datetime.now()
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', clf)
        ])

        # Perform cross-validation
        scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5, n_jobs=-1)
        precision = cross_val_score(pipeline, X_train, y_train, scoring=scoring['precision'], cv=5, n_jobs=-1)
        recall = cross_val_score(pipeline, X_train, y_train, scoring=scoring['recall'], cv=5, n_jobs=-1)
        f1 = cross_val_score(pipeline, X_train, y_train, scoring=scoring['f1'], cv=5, n_jobs=-1)

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        predictions[label] = y_pred

        cv_time = str(datetime.datetime.now() - start)[:-3]

        result_dict = {
            'Model': label,
            'Train Accuracy': pipeline.score(X_train, y_train),
            'Test Accuracy': pipeline.score(X_test, y_test),
            'Max CrossVal': scores.max(),
            'Precision': precision.mean(),
            'Recall': recall.mean(),
            'F1 Score': f1.mean(),
            'Timespan': cv_time
        }
        results.append(result_dict)

    output = pd.DataFrame(results)
    output = output.sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)

    return output, predictions

if __name__ == '__main__':

    # Load the IRIS CSV file into a DataFrame
    url = "https://raw.githubusercontent.com/matthewshawnkehoe/Iris-Classification/main/IRIS.csv"
    iris_df = pd.read_csv(url)

    # Create an SQLite engine
    engine = create_engine('sqlite:///iris.db')

    # Load the DataFrame into the SQLite database
    iris_df.to_sql('iris', engine, index=False, if_exists='replace')

    # Calculate the average sepal length
    avg_sepal_length = iris_df['sepal_length'].mean()

    # Fix row 22 which has an invalid sepal length
    iris_df = iris_df.apply(shift_row, axis=1)

    # Specify the features to convert
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']

    # Convert all feature columns to numeric
    iris_df = convert_columns_to_numeric(iris_df, features)

    # Preprocess data

    # Encode the species column into numerical values
    le = LabelEncoder()
    iris_df['species'] = le.fit_transform(iris_df['species'].astype(str))

    # Convert the species column to numerical values
    iris_df['species'] = le.fit_transform(iris_df['species'].astype(str))

    # Model Preparation

    X = iris_df.drop('species', axis=1)
    y = iris_df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



    Ellipsis