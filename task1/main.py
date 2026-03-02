import logging
from pathlib import Path

import pandas as pd

from src.preprocessing import preprocess_data
from src.eda import generate_eda_plots
from src.modeling import train_models
from src.reporting import create_pdf_report

BASE_DIR = Path(__file__).resolve().parent

# Change to 'Sales' or 'OutletSales' if a dedicated sales column is available
TARGET_COLUMN = 'MRP'


def setup_logging():
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=str(log_dir / "pipeline.log"),
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )


def main():
    setup_logging()
    logging.info("Pipeline started")

    # Ensure output directories exist
    (BASE_DIR / "reports" / "eda_plots").mkdir(parents=True, exist_ok=True)
    (BASE_DIR / "reports").mkdir(parents=True, exist_ok=True)

    try:
        data_path = BASE_DIR / "data" / "dataanalysis.xlsx"
        df = pd.read_excel(data_path)
        logging.info(f"Data loaded with shape: {df.shape}")

        df_clean = preprocess_data(df)
        logging.info("Preprocessing completed")

        if df_clean.empty:
            logging.error("Preprocessing returned an empty DataFrame. Aborting.")
            return

        output_dir = BASE_DIR / "reports" / "eda_plots"
        generate_eda_plots(df_clean, output_dir=output_dir)
        logging.info("EDA plots generated")

        rf, xgb = train_models(df_clean, target_column=TARGET_COLUMN,
                               output_dir=output_dir)
        logging.info("Modeling completed")

        create_pdf_report(base_dir=BASE_DIR)
        logging.info("PDF report generated")

        logging.info("Pipeline finished successfully")

    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}")
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
