import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sqlalchemy import create_engine
import seaborn as sns
import datetime

# Preprocess data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# Feature selection
from sklearn.decomposition import PCA

# Machine Learning
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC
import xgboost as xgb

# Read the full contents of the dataframes
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

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
            n_estimators=10, learning_rate=0.6, algorithm='SAMME'),
        'Random Forest (Ensemble)': RandomForestClassifier(n_estimators=30, max_depth=8),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        'Voting Classifier (Ensemble)': VotingClassifier(estimators=[
            ('lr', LogisticRegression(max_iter=1000)),
            ('rf', RandomForestClassifier(n_estimators=30, max_depth=8)),
            ('svm', LinearSVC(max_iter=10000)),
            ('xgb', xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
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
        scores = cross_val_score(pipeline, X_train, y_train, scoring='accuracy', cv=5, n_jobs=1)
        precision = cross_val_score(pipeline, X_train, y_train, scoring=scoring['precision'], cv=5, n_jobs=1)
        recall = cross_val_score(pipeline, X_train, y_train, scoring=scoring['recall'], cv=5, n_jobs=1)
        f1 = cross_val_score(pipeline, X_train, y_train, scoring=scoring['f1'], cv=5, n_jobs=1)

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

def pca_dimensionality_reduction(iris, X, y):
    """
    Perform PCA for dimensionality reduction on the Iris dataset and evaluate classifiers.

    Parameters:
    - iris: pd.DataFrame
        DataFrame containing the Iris features and flower species.
    - X: pd.DataFrame
        Iris features.
    - y: pd.Series
        Iris labels.

    Returns:
    - output_pca: pd.DataFrame
        DataFrame containing the performance metrics of each classifier after PCA.
    - predictions_pca: dict
        Dictionary containing the predictions of each classifier after PCA.
    """

    # Apply PCA
    pca = PCA(n_components=0.95)  # Preserve 95% of the variance
    pca_features = pca.fit_transform(X)

    print("PCA Components Shape:", pca_features.shape)
    print("Explained Variance:", pca.explained_variance_)
    print("Explained Variance Ratio:", pca.explained_variance_ratio_)

    # Split the dataset into training and testing sets
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(pca_features, y, test_size=0.25, random_state=42)

    # Convert arrays to DataFrames
    X_train_pca_df = pd.DataFrame(X_train_pca)
    X_test_pca_df = pd.DataFrame(X_test_pca)

    # Evaluate classifiers on the PCA-transformed data
    output_pca, predictions_pca = evaluate_classifiers(X_train_pca_df, X_test_pca_df, y_train_pca, y_test_pca)

    return output_pca, predictions_pca

if __name__ == '__main__':

    # Load the IRIS dataset into a Pandas Dataframe
    url = "https://raw.githubusercontent.com/matthewshawnkehoe/Iris-Classification/main/IRIS.csv"
    iris_df = pd.read_csv(url)
    engine = create_engine('sqlite:///iris.db')
    iris_df.to_sql('iris', engine, index=False, if_exists='replace')

    # Convert all feature columns to the numerical type and drop any rows with null values
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    iris_df = convert_columns_to_numeric(iris_df, features)
    iris_df = iris_df.dropna()

    # Encode the species column into numerical values
    le = LabelEncoder()
    iris_df['species'] = le.fit_transform(iris_df['species'].astype(str))

    # Prepare the model
    X = iris_df.drop('species', axis=1)
    y = iris_df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Evaluate the model
    output, predictions = evaluate_classifiers(X_train, X_test, y_train, y_test)
    print('Results of ML algorithms without dimensionality reduction: \n')
    print(output)
    print('\nThe test accuracy for all of our models is 1 (100%). Therefore, we will need to reduce overfitting in ' \
          'the test data.\n')

    # Evaluate the model with dimensionality reduction
    output_pca, predictions_pca = pca_dimensionality_reduction(iris_df, X, y)
    print('\n')
    print('Results of ML algorithms after dimensionality reduction: \n')
    print(output_pca)

    # Evaluate the model with two most important features: pedal width and length
    X_reduced = iris_df[['petal_length', 'petal_width']]
    y = iris_df['species']
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(X_reduced, y, test_size=0.25,
                                                                                        random_state=42)

    output_reduced, predictions_reduced = evaluate_classifiers(X_train_reduced, X_test_reduced, y_train_reduced,
                                                               y_test_reduced)
    print('\n')
    print('Results of ML algorithms for two most important features - pedal width and length: \n')
    print(output_pca)
