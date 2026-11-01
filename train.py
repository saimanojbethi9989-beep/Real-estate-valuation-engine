import joblib
from data_pipeline import build_preprocessor, engineer_features, load_data
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


def train():
    print("1. Loading raw property data...")
    X_raw, y = load_data()

    print("2. Engineering custom features...")
    X = engineer_features(X_raw)
    feature_names = list(X.columns)

    # Split into 80% training data and 20% testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("3. Building ML Pipeline & Training XGBoost...")
    preprocessor = build_preprocessor(feature_names)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                XGBRegressor(
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=5,
                    random_state=42,
                ),
            ),
        ]
    )

    # Train model
    model.fit(X_train, y_train)

    print("4. Evaluating on unseen test data...")
    predictions = model.predict(X_test)
    rmse = root_mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)

    print(f"-> Test RMSE: ${rmse:,.2f}")
    print(f"-> Test MAE:  ${mae:,.2f}")

    # Save trained model and feature list to disk
    joblib.dump(model, "trained_model.joblib")
    joblib.dump(feature_names, "feature_names.joblib")
    print(
        "\nSuccess: 'trained_model.joblib' has been created and saved to disk!"
    )


if __name__ == "__main__":
    train()