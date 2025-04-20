from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import shap
import matplotlib.pyplot as plt

def train_models(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    print(f"RandomForest RMSE: {mean_squared_error(y_test, rf_pred, squared=False):.2f}, R²: {r2_score(y_test, rf_pred):.3f}")

    # Train XGBoost
    xgb = XGBRegressor(n_estimators=100)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print(f"XGBoost RMSE: {mean_squared_error(y_test, xgb_pred, squared=False):.2f}, R²: {r2_score(y_test, xgb_pred):.3f}")

    # SHAP for XGBoost
    explainer = shap.Explainer(xgb)
    shap_values = explainer(X_train)
    shap.summary_plot(shap_values, X_train, show=False)
    plt.savefig("reports/eda_plots/shap_summary.png")
    plt.close()

    return rf, xgb
