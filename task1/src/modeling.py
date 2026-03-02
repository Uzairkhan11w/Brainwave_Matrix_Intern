from pathlib import Path

import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def train_models(df, target_column, output_dir=Path("reports/eda_plots")):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"RandomForest RMSE: {root_mean_squared_error(y_test, rf_pred):.2f}, R²: {r2_score(y_test, rf_pred):.3f}")

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print(f"XGBoost RMSE: {root_mean_squared_error(y_test, xgb_pred):.2f}, R²: {r2_score(y_test, xgb_pred):.3f}")

    # SHAP for XGBoost
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig(output_dir / "shap_summary.png")
    plt.close()

    return rf, xgb
