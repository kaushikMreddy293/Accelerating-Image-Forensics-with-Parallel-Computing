import streamlit as st
import pandas as pd
import base64
import io
from PIL import Image
import matplotlib.pyplot as plt
from fpdf import FPDF
import os
from datetime import datetime

class PDFExporter:
    """
    Class for exporting analysis results to PDF format
    """
    def __init__(self, title="AI vs Real Image Detection Results"):
        self.title = title
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)
        self.pdf.add_page()
        
        # Set up title
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(0, 10, self.title, ln=True, align="C")
        self.pdf.ln(10)
        
        # Add timestamp
        self.pdf.set_font("Arial", "I", 10)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)
        self.pdf.ln(5)
    
    def add_heading(self, text, size=14, style="B"):
        """Add a heading to the PDF"""
        self.pdf.set_font("Arial", style, size)
        self.pdf.cell(0, 10, text, ln=True)
        self.pdf.ln(2)
    
    def add_text(self, text, size=12, style=""):
        """Add text to the PDF"""
        self.pdf.set_font("Arial", style, size)
        self.pdf.multi_cell(0, 10, text)
        self.pdf.ln(2)
    
    def add_image(self, image_path, caption=None, w=80):
        """Add an image to the PDF"""
        try:
            self.pdf.image(image_path, x=None, y=None, w=w)
            if caption:
                self.pdf.set_font("Arial", "I", 10)
                self.pdf.cell(0, 10, caption, ln=True, align="C")
            self.pdf.ln(5)
        except Exception as e:
            self.add_text(f"Error adding image: {str(e)}")
    
    def add_image_from_pil(self, pil_image, caption=None, w=80):
        """Add a PIL Image to the PDF"""
        try:
            # Save PIL image to a temporary file
            temp_path = "temp_pdf_image.jpg"
            pil_image.save(temp_path)
            
            # Add the image to the PDF
            self.pdf.image(temp_path, x=None, y=None, w=w)
            
            # Add caption if provided
            if caption:
                self.pdf.set_font("Arial", "I", 10)
                self.pdf.cell(0, 10, caption, ln=True, align="C")
            
            self.pdf.ln(5)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            self.add_text(f"Error adding image: {str(e)}")
    
    def add_table(self, headers, data):
        """Add a table to the PDF"""
        # Set up table
        self.pdf.set_font("Arial", "B", 12)
        
        # Calculate column width
        col_width = self.pdf.w / len(headers)
        
        # Add headers
        for header in headers:
            self.pdf.cell(col_width, 10, header, border=1)
        self.pdf.ln()
        
        # Add data
        self.pdf.set_font("Arial", "", 12)
        for row in data:
            for item in row:
                self.pdf.cell(col_width, 10, str(item), border=1)
            self.pdf.ln()
        
        self.pdf.ln(5)
    
    def add_dataframe(self, df):
        """Add a pandas DataFrame to the PDF"""
        headers = df.columns.tolist()
        data = df.values.tolist()
        self.add_table(headers, data)
    
    def add_chart(self, fig, caption=None, w=160):
        """Add a matplotlib figure to the PDF"""
        try:
            # Save figure to a temporary file
            temp_path = "temp_chart.png"
            fig.savefig(temp_path, bbox_inches="tight")
            
            # Add the image to the PDF
            self.pdf.image(temp_path, x=None, y=None, w=w)
            
            # Add caption if provided
            if caption:
                self.pdf.set_font("Arial", "I", 10)
                self.pdf.cell(0, 10, caption, ln=True, align="C")
            
            self.pdf.ln(5)
            
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            self.add_text(f"Error adding chart: {str(e)}")
    
    def add_page_break(self):
        """Add a page break to the PDF"""
        self.pdf.add_page()
    
    def get_pdf_bytes(self):
        """Get the PDF as bytes"""
        return self.pdf.output(dest="S").encode("latin1")
    
    def save(self, output_path):
        """Save the PDF to a file"""
        self.pdf.output(output_path)


def export_single_analysis_to_pdf(image, prediction, model_name, heatmap_fig=None):
    """
    Export a single analysis result to PDF
    
    Args:
        image: PIL Image object
        prediction: Prediction dictionary
        model_name: Name of the model used
        heatmap_fig: Optional matplotlib figure with heatmap visualization
        
    Returns:
        PDF bytes
    """
    # Create PDF
    pdf_exporter = PDFExporter(title="AI vs Real Image Detection - Single Analysis")
    
    # Add model information
    pdf_exporter.add_heading(f"Model: {model_name}")
    
    # Add prediction results
    result_class = prediction['class']
    confidence = prediction['confidence'] * 100
    pdf_exporter.add_text(f"Prediction: {result_class}")
    pdf_exporter.add_text(f"Confidence: {confidence:.2f}%")
    pdf_exporter.add_text(f"Processing Time: {prediction.get('processing_time_ms', 0):.2f} ms")
    
    # Add confidence scores
    pdf_exporter.add_heading("Confidence Scores", size=12)
    for cls, prob in prediction['probabilities'].items():
        pdf_exporter.add_text(f"{cls}: {prob*100:.2f}%")
    
    # Add image
    pdf_exporter.add_heading("Analyzed Image", size=12)
    pdf_exporter.add_image_from_pil(image, caption="Uploaded Image", w=120)
    
    # Add heatmap if available
    if heatmap_fig:
        pdf_exporter.add_heading("Grad-CAM Visualization", size=12)
        pdf_exporter.add_chart(heatmap_fig, caption="Regions influencing the model's decision", w=160)
    
    # Add explanation
    pdf_exporter.add_heading("Explanation", size=12)
    if result_class == "Real":
        explanation = f"The {model_name} model detected natural patterns, lighting, and textures consistent with real photographs. The image shows natural perspective and depth that AI models often struggle to replicate perfectly."
    else:
        explanation = f"The {model_name} model detected unnatural patterns, inconsistent lighting, or unusual textures that are common in AI-generated images. Look for unusual details in faces, hands, or backgrounds."
    pdf_exporter.add_text(explanation)
    
    # Return PDF bytes
    return pdf_exporter.get_pdf_bytes()


def export_batch_results_to_pdf(batch_results):
    """
    Export batch analysis results to PDF
    
    Args:
        batch_results: List of batch result dictionaries
        
    Returns:
        PDF bytes
    """
    # Create PDF
    pdf_exporter = PDFExporter(title="AI vs Real Image Detection - Batch Analysis")
    
    # Add summary
    pdf_exporter.add_heading("Summary")
    total_images = len(batch_results)
    real_count = sum(1 for r in batch_results if r["prediction"] == "Real")
    ai_count = sum(1 for r in batch_results if r["prediction"] == "AI-generated")
    pdf_exporter.add_text(f"Total Images: {total_images}")
    pdf_exporter.add_text(f"Real Images: {real_count}")
    pdf_exporter.add_text(f"AI-Generated Images: {ai_count}")
    
    # Create summary chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Real', 'AI-Generated']
    sizes = [real_count, ai_count]
    colors = ['#4CAF50', '#F44336']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Distribution of Predictions')
    
    # Add chart to PDF
    pdf_exporter.add_chart(fig, caption="Distribution of Predictions")
    
    # Add results table
    pdf_exporter.add_heading("Detailed Results")
    
    # Create DataFrame for table
    df = pd.DataFrame([
        {
            "Filename": r["filename"],
            "Prediction": r["prediction"],
            "Confidence": f"{r['confidence']:.2f}%",
            "Processing Time": f"{r['processing_time_ms']:.2f} ms"
        } for r in batch_results
    ])
    
    # Add table to PDF
    pdf_exporter.add_dataframe(df)
    
    # Add gallery of images
    pdf_exporter.add_heading("Image Gallery")
    pdf_exporter.add_text("Sample of analyzed images with predictions:")
    
    # Add up to 9 images (3 per row, 3 rows)
    images_to_show = min(9, len(batch_results))
    for i in range(0, images_to_show, 3):
        for j in range(min(3, images_to_show - i)):
            result = batch_results[i + j]
            try:
                img = Image.open(io.BytesIO(base64.b64decode(result["image_data"])))
                caption = f"{result['filename']}: {result['prediction']} ({result['confidence']:.2f}%)"
                pdf_exporter.add_image_from_pil(img, caption=caption, w=60)
            except Exception as e:
                pdf_exporter.add_text(f"Error displaying image: {str(e)}")
        
        # Add some space between rows
        pdf_exporter.add_text("")
    
    # Return PDF bytes
    return pdf_exporter.get_pdf_bytes()


def export_history_to_pdf(history):
    """
    Export analysis history to PDF
    
    Args:
        history: List of history dictionaries
        
    Returns:
        PDF bytes
    """
    # Create PDF
    pdf_exporter = PDFExporter(title="AI vs Real Image Detection - Analysis History")
    
    # Add summary
    pdf_exporter.add_heading("Summary")
    total_analyses = len(history)
    real_count = sum(1 for h in history if h["prediction"] == "Real")
    ai_count = sum(1 for h in history if h["prediction"] == "AI-generated")
    pdf_exporter.add_text(f"Total Analyses: {total_analyses}")
    pdf_exporter.add_text(f"Real Images: {real_count}")
    pdf_exporter.add_text(f"AI-Generated Images: {ai_count}")
    
    # Create summary chart
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['Real', 'AI-Generated']
    sizes = [real_count, ai_count]
    colors = ['#4CAF50', '#F44336']
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    plt.title('Distribution of Predictions')
    
    # Add chart to PDF
    pdf_exporter.add_chart(fig, caption="Distribution of Predictions")
    
    # Add results table
    pdf_exporter.add_heading("Analysis History")
    
    # Create DataFrame for table
    df = pd.DataFrame([
        {
            "Timestamp": h["timestamp"],
            "Model": h["model"],
            "Prediction": h["prediction"],
            "Confidence": f"{h['confidence']:.2f}%"
        } for h in history
    ])
    
    # Add table to PDF
    pdf_exporter.add_dataframe(df)
    
    # Add gallery of images
    pdf_exporter.add_heading("Image Gallery")
    pdf_exporter.add_text("Sample of analyzed images with predictions:")
    
    # Add up to 9 images (3 per row, 3 rows)
    images_to_show = min(9, len(history))
    for i in range(0, images_to_show, 3):
        for j in range(min(3, images_to_show - i)):
            item = history[i + j]
            try:
                img = Image.open(io.BytesIO(base64.b64decode(item["image_data"])))
                caption = f"{item['model']}: {item['prediction']} ({item['confidence']:.2f}%)"
                pdf_exporter.add_image_from_pil(img, caption=caption, w=60)
            except Exception as e:
                pdf_exporter.add_text(f"Error displaying image: {str(e)}")
        
        # Add some space between rows
        pdf_exporter.add_text("")
    
    # Return PDF bytes
    return pdf_exporter.get_pdf_bytes()


def get_download_link(binary_data, filename, text):
    """
    Generate a download link for binary data
    
    Args:
        binary_data: Binary data to download
        filename: Name of the file to download
        text: Text to display for the download link
        
    Returns:
        HTML string with download link
    """
    b64 = base64.b64encode(binary_data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'
    return href
