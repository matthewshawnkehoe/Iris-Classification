import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.main import evaluate_classifiers, convert_columns_to_numeric

@pytest.fixture
def iris_data():
    url = "https://raw.githubusercontent.com/matthewshawnkehoe/Iris-Classification/main/IRIS.csv"
    iris_df = pd.read_csv(url)
    features = ['petal_length', 'petal_width', 'sepal_length', 'sepal_width']
    iris_df = convert_columns_to_numeric(iris_df, features)
    iris_df = iris_df.dropna()
    le = LabelEncoder()
    iris_df['species'] = le.fit_transform(iris_df['species'].astype(str))
    X = iris_df.drop('species', axis=1)
    y = iris_df['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

def test_evaluate_classifiers(iris_data):
    X_train, X_test, y_train, y_test = iris_data
    output, predictions = evaluate_classifiers(X_train, X_test, y_train, y_test)
    assert output is not None
    assert len(predictions) > 0

def test_convert_columns_to_numeric():
    df = pd.DataFrame({'A': ['1.0', '2.1', '3.2'], 'B': ['4.4', '5.4', '6.0']})
    columns = ['A', 'B']
    converted_df = convert_columns_to_numeric(df, columns)
    assert converted_df['A'].dtype == 'float64'
    assert converted_df['B'].dtype == 'float64'
