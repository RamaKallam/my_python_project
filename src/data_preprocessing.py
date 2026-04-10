import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(df: pd.DataFrame):
    """
    Splits features and target from dataframe.
    """
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits dataset into train and test sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler