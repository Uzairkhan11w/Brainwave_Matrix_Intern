import matplotlib.pyplot as plt
import seaborn as sns

def generate_eda_plots(df):
    numeric_features = ['MRP', 'ProductVisibility', 'Weight']
    
    for feature in numeric_features:
        plt.figure(figsize=(6,4))
        sns.histplot(df[feature].dropna(), kde=True)
        plt.title(f"Distribution of {feature}")
        plt.savefig(f"reports/eda_plots/{feature}_distribution.png")
        plt.close()

    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlation Heatmap")
    plt.savefig("reports/eda_plots/correlation_heatmap.png")
    plt.close()

    if 'OutletType_Supermarket Type1' in df.columns:
        df['OutletType'] = df.apply(
            lambda row: 'Supermarket Type1' if row.get('OutletType_Supermarket Type1', 0) == 1 else 'Grocery Store',
            axis=1
        )
        plt.figure(figsize=(6,4))
        sns.boxplot(x='OutletType', y='MRP', data=df)
        plt.title("MRP by Outlet Type")
        plt.savefig("reports/eda_plots/MRP_by_OutletType.png")
        plt.close()
