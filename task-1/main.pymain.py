import logging
from src.preprocessing import preprocess_data
from src.eda import generate_eda_plots
from src.modeling import train_models
from src.reporting import create_pdf_report
import pandas as pd

def setup_logging():
    logging.basicConfig(
        filename='logs/pipeline.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )

def main():
    setup_logging()
    logging.info("Pipeline started")

    df = pd.read_excel("data/dataanalysis.xlsx")
    logging.info(f"Data loaded with shape: {df.shape}")

    df_clean = preprocess_data(df)
    logging.info("Preprocessing completed")

    generate_eda_plots(df_clean)
    logging.info("EDA plots generated")

    rf, xgb = train_models(df_clean, target_column='MRP')
    logging.info("Modeling completed")

    create_pdf_report()
    logging.info("PDF report generated")

    logging.info("Pipeline finished successfully")

if __name__ == "__main__":
    main()
