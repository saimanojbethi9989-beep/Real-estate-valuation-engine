import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


def load_data():
    """Fetches real housing data and converts prices into whole dollar values."""
    raw = fetch_california_housing(as_frame=True)
    df = raw.frame

    # Target price is converted from hundreds of thousands into real dollars
    y = df["MedHouseVal"] * 100000
    X = df.drop(columns=["MedHouseVal"])
    return X, y


def engineer_features(df):
    """Calculates custom spatial and room density metrics."""
    df = df.copy()

    # Prevent division by zero
    safe_rooms = np.where(df["AveRooms"] == 0, 1e-5, df["AveRooms"])

    # Feature 1: Ratio of bedrooms compared to total rooms
    df["BedroomsRatio"] = df["AveBedrms"] / safe_rooms

    # Feature 2: Population density per room
    df["DensityPerRoom"] = df["Population"] / (safe_rooms + 1)

    return df


def build_preprocessor(feature_columns):
    """Creates a transformer that cleans and scales numeric data."""
    transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("numeric", transformer, feature_columns)]
    )