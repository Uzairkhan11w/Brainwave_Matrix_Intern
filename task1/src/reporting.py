from pathlib import Path

from fpdf import FPDF


def create_pdf_report(base_dir=Path(".")):
    base_dir = Path(base_dir)
    plots_dir = base_dir / "reports" / "eda_plots"
    output_path = base_dir / "reports" / "sales_analysis_report.pdf"
    (base_dir / "reports").mkdir(parents=True, exist_ok=True)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Sales Analysis Report", ln=True, align='C')
    pdf.ln(10)

    # Section 1: MRP Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "1. MRP Distribution", ln=True)
    mrp_plot = plots_dir / "MRP_distribution.png"
    if mrp_plot.exists():
        pdf.image(str(mrp_plot), w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "MRP distribution plot not found.", ln=True)
    pdf.ln(10)

    # Section 2: Product Visibility Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "2. Product Visibility", ln=True)
    vis_plot = plots_dir / "ProductVisibility_distribution.png"
    if vis_plot.exists():
        pdf.image(str(vis_plot), w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "Product Visibility plot not found.", ln=True)
    pdf.ln(10)

    # Section 3: Weight Distribution
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "3. Weight Distribution", ln=True)
    weight_plot = plots_dir / "Weight_distribution.png"
    if weight_plot.exists():
        pdf.image(str(weight_plot), w=180)
    else:
        pdf.set_font("Arial", '', 10)
        pdf.cell(0, 10, "Weight plot not found.", ln=True)
    pdf.ln(10)

    # Save the PDF
    pdf.output(str(output_path))
    print(f"✅ PDF report saved to: {output_path}")
