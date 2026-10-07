import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def train_price_model(train_path='data/train.csv', model_save_path='models/property_price_model.pkl'):
    df = pd.read_csv(train_path)

    # Key features selected for realistic real estate valuation
    features = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'FullBath', 'BedroomAbvGr', 'GarageCars', 'Neighborhood']
    target = 'SalePrice'

    X = df[features]
    y = df[target]

    num_cols = ['GrLivArea', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'FullBath', 'BedroomAbvGr', 'GarageCars']
    cat_cols = ['Neighborhood']

    # Preprocessing pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)

    rmse = np.sqrt(mean_squared_error(y_val, preds))
    r2 = r2_score(y_val, preds)

    print(f"Model Training Completed.")
    print(f"Validation RMSE: ${rmse:.2f}")
    print(f"Validation R2 Score: {r2:.4f}")

    joblib.dump(model, model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_price_model()