from fpdf import FPDF
import os

def create_pdf_report():
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Sales Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Section 1: MRP Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. MRP Distribution", ln=True)
    if os.path.exists("reports/eda_plots/MRP_distribution.png"):
        pdf.image("reports/eda_plots/MRP_distribution.png", w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "MRP distribution plot not found.", ln=True)
    pdf.ln(10)

    # Section 2: Product Visibility Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Product Visibility", ln=True)
    if os.path.exists("reports/eda_plots/ProductVisibility_distribution.png"):
        pdf.image("reports/eda_plots/ProductVisibility_distribution.png", w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "Product Visibility plot not found.", ln=True)
    pdf.ln(10)

    # Section 3: Weight Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Weight Distribution", ln=True)
    if os.path.exists("reports/eda_plots/Weight_distribution.png"):
        pdf.image("reports/eda_plots/Weight_distribution.png", w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "Weight plot not found.", ln=True)
    pdf.ln(10)

    # Save the PDF
    output_path = "reports/sales_analysis_report.pdf"
    pdf.output(output_path)
    print(f"✅ PDF report saved to: {output_path}")
