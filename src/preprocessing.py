import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

def preprocess_data(df: pd.DataFrame, use_knn=False) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

    if use_knn:
        knn_imputer = KNNImputer(n_neighbors=5)
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        df[numeric_cols] = knn_imputer.fit_transform(df[numeric_cols])
    else:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        cat_imputer = SimpleImputer(strategy='most_frequent')
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Encode 'OutletSize' as ordinal
    if 'OutletSize' in df.columns:
        size_order = {'Small': 0, 'Medium': 1, 'High': 2}
        df['OutletSize'] = df['OutletSize'].map(size_order)

    # One-hot encode remaining categoricals
    nominal_cols = [col for col in categorical_cols if col != 'OutletSize']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=False)

    # Safely filter by OutletType and LocationType
    outlet_cols = df.columns[df.columns.str.startswith('OutletType_')].tolist()
    location_cols = df.columns[df.columns.str.startswith('LocationType_')].tolist()

    filter_condition = pd.Series([False] * len(df))
    if 'OutletType_Supermarket Type1' in outlet_cols:
        filter_condition |= df['OutletType_Supermarket Type1'] == 1
    if 'OutletType_Grocery Store' in outlet_cols:
        filter_condition |= df['OutletType_Grocery Store'] == 1

    df = df[filter_condition]

    if 'LocationType_Tier 2' in location_cols:
        df = df[df['LocationType_Tier 2'] == 1]

    return df
