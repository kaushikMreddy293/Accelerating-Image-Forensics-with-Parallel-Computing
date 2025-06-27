import streamlit as st
import torch
import os
import io
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from models import ResNet18Model, VisionTransformerModel
import pandas as pd
import base64
from datetime import datetime
from pdf_exporter import export_single_analysis_to_pdf, export_batch_results_to_pdf, export_history_to_pdf, get_download_link

# Set page configuration
st.set_page_config(
    page_title="AI vs Real Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling and centered layout
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
        text-align: center;
    }
    .result-real {
        color: #4CAF50;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .result-ai {
        color: #F44336;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .model-info {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .stButton button {
        background-color: #1E88E5;
        color: white;
    }
    .confidence-bar {
        margin-top: 10px;
        margin-bottom: 20px;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
    }
    .metric-item {
        text-align: center;
        padding: 10px;
        background-color: #f8f9fa;
        border-radius: 5px;
        flex: 1;
        margin: 0 5px;
    }
    .stProgress > div > div {
        background-color: var(--bar-color, #1E88E5);
    }
    /* Center the main content */
    .block-container {
        max-width: 1000px;
        padding-top: 2rem;
        padding-bottom: 2rem;
        margin: 0 auto;
    }
    /* Gallery styling */
    .gallery-item {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px;
        background-color: #f9f9f9;
    }
    .gallery-image {
        width: 100%;
        height: auto;
        border-radius: 5px;
    }
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'resnet_model' not in st.session_state:
    st.session_state.resnet_model = None
if 'vit_model' not in st.session_state:
    st.session_state.vit_model = None
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Function to load models
@st.cache_resource
def load_models():
    try:
        resnet_model = ResNet18Model()
        vit_model = VisionTransformerModel()
        return resnet_model, vit_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Create a container for the main content
main_container = st.container()

with main_container:
    # Header section
    st.markdown("<h1 class='main-header'>AI vs Real Image Detection</h1>", unsafe_allow_html=True)
    st.markdown("""
    <p style='text-align: center;'>
    This application helps you detect whether an image is AI-generated or real using state-of-the-art 
    deep learning models. Upload your image and select which model(s) you'd like to use for analysis.
    </p>
    """, unsafe_allow_html=True)

    # Create tabs for different functionalities
    tabs = st.tabs(["Single Image Analysis", "Batch Processing", "History & Gallery", "Export Results"])
    
    # Tab 1: Single Image Analysis
    with tabs[0]:
        # Create a placeholder for the uploaded image
        uploaded_image_placeholder = st.empty()

        # Image upload section
        st.markdown("<h2 class='sub-header'>Upload an Image</h2>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], key="single_upload")

        # Sample images option
        use_sample = st.checkbox("Or use a sample image")
        sample_image_path = None

        # Create sample directory if it doesn't exist
        sample_dir = os.path.join(os.getcwd(), "samples")
        os.makedirs(sample_dir, exist_ok=True)

        # Create a default sample image if none exists
        if not os.path.exists(os.path.join(sample_dir, "sample_real.jpg")):
            # Create a simple colored image as a placeholder
            sample_img = Image.new('RGB', (224, 224), color=(73, 109, 137))
            sample_img.save(os.path.join(sample_dir, "sample_real.jpg"))

        if use_sample:
            sample_image_path = os.path.join(sample_dir, "sample_real.jpg")
            if os.path.exists(sample_image_path):
                image = Image.open(sample_image_path)
                uploaded_image_placeholder.image(image, caption="Sample Image", use_column_width=True)
            else:
                st.warning("Sample image not found. Please upload your own image.")
                use_sample = False

        # Display the uploaded image
        image = None
        if uploaded_file is not None:
            try:
                # Read the image file
                image_bytes = uploaded_file.getvalue()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Display the image
                uploaded_image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Save the uploaded image temporarily
                temp_image_path = os.path.join(os.getcwd(), "temp_image.jpg")
                image.save(temp_image_path)
            except Exception as e:
                st.error(f"Error processing uploaded image: {str(e)}")
        elif use_sample and os.path.exists(sample_image_path):
            image = Image.open(sample_image_path)

        # Model selection section
        st.markdown("<h2 class='sub-header'>Select Model</h2>", unsafe_allow_html=True)
        model_option = st.radio(
            "Choose which model(s) to use:",
            ("ResNet18", "Vision Transformer (ViT)", "Compare Both")
        )

        # Add visualization options
        show_explanation = st.checkbox("Show Explainable AI Features", value=False)

        # Add an analyze button
        analyze_button = st.button("Analyze Image")

        # Function to provide explainable AI features
        def generate_explanation(model_name, prediction):
            """Generate explanation for the model's decision"""
            explanation = {
                "ResNet18": {
                    "Real": "The ResNet18 model detected natural patterns, lighting, and textures consistent with real photographs. The image shows natural perspective and depth that AI models often struggle to replicate perfectly.",
                    "AI-generated": "The ResNet18 model detected unnatural patterns, inconsistent lighting, or unusual textures that are common in AI-generated images. Look for unusual details in faces, hands, or backgrounds."
                },
                "Vision Transformer": {
                    "Real": "The Vision Transformer model analyzed the image patches and found consistent patterns across different regions. The global coherence of the image suggests it was captured by a camera rather than generated.",
                    "AI-generated": "The Vision Transformer model detected inconsistencies between image patches that are typical of AI generation. The model found patterns that don't occur naturally in photographs."
                }
            }
            
            return explanation[model_name][prediction['class']]

        # Function to display prediction results
        def display_prediction(prediction, model_name):
            """Display prediction results with styling"""
            if prediction is None:
                st.error(f"Error: {model_name} prediction failed")
                return
            
            result_class = prediction['class']
            confidence = prediction['confidence'] * 100
            
            # Display model info
            st.markdown(f"<div class='model-info'>Model: {model_name}<br>Processing Time: {prediction.get('processing_time_ms', 0):.2f} ms</div>", unsafe_allow_html=True)
            
            # Display prediction with appropriate styling
            if result_class == "Real":
                st.markdown(f"<p class='result-real'>Prediction: {result_class} ({confidence:.2f}% confidence)</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p class='result-ai'>Prediction: {result_class} ({confidence:.2f}% confidence)</p>", unsafe_allow_html=True)
            
            # Display confidence bars for both classes
            st.markdown("<p>Confidence Scores:</p>", unsafe_allow_html=True)
            for cls, prob in prediction['probabilities'].items():
                bar_color = "#4CAF50" if cls == "Real" else "#F44336"
                st.markdown(f"<p>{cls}: {prob*100:.2f}%</p>", unsafe_allow_html=True)
                st.markdown(f"""
                <style>
                [data-testid="stProgress"] {{
                    --bar-color: {bar_color};
                }}
                </style>
                """, unsafe_allow_html=True)
                st.progress(prob)
            
            # Display explainable AI features if selected
            if show_explanation:
                st.markdown("### Explanation")
                explanation = generate_explanation(model_name, prediction)
                st.info(explanation)
            
            # Add to history
            if model_name not in ["ResNet18 (Batch)", "Vision Transformer (Batch)"]:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Save the image if it's not already in history
                img_copy = image.copy()
                img_buffer = io.BytesIO()
                img_copy.save(img_buffer, format="JPEG")
                img_str = base64.b64encode(img_buffer.getvalue()).decode()
                
                history_item = {
                    "timestamp": timestamp,
                    "model": model_name,
                    "prediction": result_class,
                    "confidence": confidence,
                    "image_data": img_str
                }
                st.session_state.history.append(history_item)

        # Function to create and display a comparison chart
        def display_comparison_chart(resnet_pred, vit_pred):
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Data for the chart
            models = ['ResNet18', 'Vision Transformer']
            real_probs = [resnet_pred['probabilities']['Real'], vit_pred['probabilities']['Real']]
            ai_probs = [resnet_pred['probabilities']['AI-generated'], vit_pred['probabilities']['AI-generated']]
            
            x = np.arange(len(models))
            width = 0.35
            
            # Create the bars
            ax.bar(x - width/2, [p*100 for p in real_probs], width, label='Real', color='#4CAF50', alpha=0.7)
            ax.bar(x + width/2, [p*100 for p in ai_probs], width, label='AI-generated', color='#F44336', alpha=0.7)
            
            # Add labels and title
            ax.set_ylabel('Confidence (%)')
            ax.set_title('Model Comparison: Confidence Scores')
            ax.set_xticks(x)
            ax.set_xticklabels(models)
            ax.legend()
            
            # Add value labels on bars
            for i, v in enumerate(real_probs):
                ax.text(i - width/2, v*100 + 1, f'{v*100:.1f}%', ha='center', fontsize=9)
            for i, v in enumerate(ai_probs):
                ax.text(i + width/2, v*100 + 1, f'{v*100:.1f}%', ha='center', fontsize=9)
            
            ax.set_ylim(0, 105)  # Set y-axis limit to accommodate labels
            
            # Display the chart
            st.pyplot(fig)

        # When the analyze button is clicked and an image is available
        if analyze_button and image is not None:
            # Load models
            with st.spinner("Loading models..."):
                resnet_model, vit_model = load_models()
            
            if resnet_model is None or vit_model is None:
                st.error("Failed to load models. Please try again.")
            else:
                # Show a spinner while processing
                with st.spinner("Analyzing image..."):
                    try:
                        # Create a two-column layout for results
                        if model_option == "Compare Both":
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("<h3>ResNet18 Results</h3>", unsafe_allow_html=True)
                                resnet_prediction = resnet_model.predict(image)
                                display_prediction(resnet_prediction, "ResNet18")
                            
                            with col2:
                                st.markdown("<h3>Vision Transformer Results</h3>", unsafe_allow_html=True)
                                vit_prediction = vit_model.predict(image)
                                display_prediction(vit_prediction, "Vision Transformer")
                            
                            # Add comparison section
                            st.markdown("<h3>Model Comparison</h3>", unsafe_allow_html=True)
                            
                            # Check if models agree
                            if resnet_prediction['class'] == vit_prediction['class']:
                                agreement = "Both models agree"
                                agreement_color = "#4CAF50"
                            else:
                                agreement = "Models disagree"
                                agreement_color = "#F44336"
                            
                            st.markdown(f"<p style='color: {agreement_color}; font-weight: bold; font-size: 1.2rem;'>{agreement}</p>", unsafe_allow_html=True)
                            
                            # Display comparison chart
                            display_comparison_chart(resnet_prediction, vit_prediction)
                            
                            # Display metrics comparison
                            st.markdown("<div class='metrics-container'>", unsafe_allow_html=True)
                            st.markdown(f"""
                                <div class='metric-item'>
                                    <h4>ResNet18</h4>
                                    <p>Confidence: {resnet_prediction['confidence']*100:.2f}%</p>
                                    <p>Processing Time: {resnet_prediction.get('processing_time_ms', 0):.2f} ms</p>
                                </div>
                                <div class='metric-item'>
                                    <h4>Vision Transformer</h4>
                                    <p>Confidence: {vit_prediction['confidence']*100:.2f}%</p>
                                    <p>Processing Time: {vit_prediction.get('processing_time_ms', 0):.2f} ms</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            
                        else:
                            # Single model result
                            st.markdown(f"<h3>{model_option} Results</h3>", unsafe_allow_html=True)
                            
                            if model_option == "ResNet18":
                                prediction = resnet_model.predict(image)
                                display_prediction(prediction, "ResNet18")
                            else:
                                prediction = vit_model.predict(image)
                                display_prediction(prediction, "Vision Transformer")
                    except Exception as e:
                        st.error(f"Error during image analysis: {str(e)}")
                        st.info("Please try with a different image or check if the models are loaded correctly.")

    # Tab 2: Batch Processing
    with tabs[1]:
        st.markdown("<h2 class='sub-header'>Batch Image Processing</h2>", unsafe_allow_html=True)
        st.markdown("Upload multiple images to analyze them in batch.")
        
        # Batch upload
        batch_files = st.file_uploader("Choose multiple image files", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_upload")
        
        # Model selection for batch processing
        batch_model = st.radio(
            "Choose which model to use for batch processing:",
            ("ResNet18", "Vision Transformer (ViT)")
        )
        
        # Process batch button
        process_batch = st.button("Process Batch")
        
        if process_batch and batch_files:
            # Load models
            with st.spinner("Loading models..."):
                resnet_model, vit_model = load_models()
            
            if (resnet_model is None and batch_model == "ResNet18") or (vit_model is None and batch_model == "Vision Transformer (ViT)"):
                st.error("Failed to load models. Please try again.")
            else:
                # Clear previous batch results
                st.session_state.batch_results = []
                
                # Process each image
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, uploaded_file in enumerate(batch_files):
                    try:
                        # Update progress
                        progress = (i + 1) / len(batch_files)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing image {i+1} of {len(batch_files)}")
                        
                        # Read the image
                        image_bytes = uploaded_file.getvalue()
                        image = Image.open(io.BytesIO(image_bytes))
                        
                        # Get prediction
                        if batch_model == "ResNet18":
                            prediction = resnet_model.predict(image)
                            model_name = "ResNet18 (Batch)"
                        else:
                            prediction = vit_model.predict(image)
                            model_name = "Vision Transformer (Batch)"
                        
                        # Save image data
                        img_buffer = io.BytesIO()
                        image.save(img_buffer, format="JPEG")
                        img_str = base64.b64encode(img_buffer.getvalue()).decode()
                        
                        # Save result
                        result = {
                            "filename": uploaded_file.name,
                            "model": batch_model,
                            "prediction": prediction['class'],
                            "confidence": prediction['confidence'] * 100,
                            "processing_time_ms": prediction.get('processing_time_ms', 0),
                            "image_data": img_str
                        }
                        st.session_state.batch_results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                
                # Complete
                progress_bar.progress(1.0)
                status_text.text(f"Completed processing {len(batch_files)} images")
                
                # Display results
                if st.session_state.batch_results:
                    st.markdown("### Batch Processing Results")
                    
                    # Create a DataFrame for display
                    df = pd.DataFrame([
                        {
                            "Filename": r["filename"],
                            "Prediction": r["prediction"],
                            "Confidence": f"{r['confidence']:.2f}%",
                            "Processing Time": f"{r['processing_time_ms']:.2f} ms"
                        } for r in st.session_state.batch_results
                    ])
                    
                    st.dataframe(df)
                    
                    # Display summary statistics
                    real_count = sum(1 for r in st.session_state.batch_results if r["prediction"] == "Real")
                    ai_count = sum(1 for r in st.session_state.batch_results if r["prediction"] == "AI-generated")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Real Images", real_count)
                    with col2:
                        st.metric("AI-Generated Images", ai_count)
                    
                    # Display gallery of processed images
                    st.markdown("### Batch Processing Gallery")
                    
                    # Create rows of 3 images each
                    for i in range(0, len(st.session_state.batch_results), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            if i+j < len(st.session_state.batch_results):
                                result = st.session_state.batch_results[i+j]
                                with cols[j]:
                                    st.markdown(f"<div class='gallery-item'>", unsafe_allow_html=True)
                                    st.image(
                                        Image.open(io.BytesIO(base64.b64decode(result["image_data"]))),
                                        caption=f"{result['filename']}: {result['prediction']} ({result['confidence']:.2f}%)",
                                        use_column_width=True
                                    )
                                    st.markdown("</div>", unsafe_allow_html=True)
    
    # Tab 3: History & Gallery
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Analysis History & Gallery</h2>", unsafe_allow_html=True)
        
        if not st.session_state.history:
            st.info("No analysis history yet. Analyze some images to see them here.")
        else:
            # Add clear history button
            if st.button("Clear History"):
                st.session_state.history = []
                st.experimental_rerun()
            
            # Display history as a gallery
            st.markdown("### Analysis Gallery")
            
            # Create rows of 3 images each
            for i in range(0, len(st.session_state.history), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i+j < len(st.session_state.history):
                        item = st.session_state.history[i+j]
                        with cols[j]:
                            st.markdown(f"<div class='gallery-item'>", unsafe_allow_html=True)
                            st.image(
                                Image.open(io.BytesIO(base64.b64decode(item["image_data"]))),
                                caption=f"{item['model']}: {item['prediction']} ({item['confidence']:.2f}%)",
                                use_column_width=True
                            )
                            st.markdown(f"<p>Analyzed on: {item['timestamp']}</p>", unsafe_allow_html=True)
                            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display history as a table
            st.markdown("### Analysis History Table")
            
            # Create a DataFrame for display
            df = pd.DataFrame([
                {
                    "Timestamp": h["timestamp"],
                    "Model": h["model"],
                    "Prediction": h["prediction"],
                    "Confidence": f"{h['confidence']:.2f}%"
                } for h in st.session_state.history
            ])
            
            st.dataframe(df)
    
    # Tab 4: Export Results
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Export Results</h2>", unsafe_allow_html=True)
        
        export_type = st.radio(
            "Choose export format:",
            ("CSV", "PDF")
        )
        
        export_data = st.radio(
            "Choose data to export:",
            ("Single Analysis", "Batch Results", "History")
        )
        
        if st.button("Generate Export"):
            if export_data == "Single Analysis" and image is not None and 'prediction' in locals():
                if export_type == "CSV":
                    # Create CSV for single analysis
                    df = pd.DataFrame([{
                        "Model": model_option,
                        "Prediction": prediction['class'],
                        "Confidence": f"{prediction['confidence']*100:.2f}%",
                        "Processing Time": f"{prediction.get('processing_time_ms', 0):.2f} ms"
                    }])
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download link
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="single_analysis.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Generate PDF for single analysis
                    try:
                        pdf_bytes = export_single_analysis_to_pdf(image, prediction, model_option)
                        download_link = get_download_link(pdf_bytes, "single_analysis.pdf", "Download PDF")
                        st.markdown(download_link, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
            
            elif export_data == "Batch Results" and st.session_state.batch_results:
                if export_type == "CSV":
                    # Create CSV for batch results
                    df = pd.DataFrame([
                        {
                            "Filename": r["filename"],
                            "Model": r["model"],
                            "Prediction": r["prediction"],
                            "Confidence": f"{r['confidence']:.2f}%",
                            "Processing Time": f"{r['processing_time_ms']:.2f} ms"
                        } for r in st.session_state.batch_results
                    ])
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download link
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="batch_results.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Generate PDF for batch results
                    try:
                        pdf_bytes = export_batch_results_to_pdf(st.session_state.batch_results)
                        download_link = get_download_link(pdf_bytes, "batch_results.pdf", "Download PDF")
                        st.markdown(download_link, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
            
            elif export_data == "History" and st.session_state.history:
                if export_type == "CSV":
                    # Create CSV for history
                    df = pd.DataFrame([
                        {
                            "Timestamp": h["timestamp"],
                            "Model": h["model"],
                            "Prediction": h["prediction"],
                            "Confidence": f"{h['confidence']:.2f}%"
                        } for h in st.session_state.history
                    ])
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Create download link
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="analysis_history.csv">Download CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                else:
                    # Generate PDF for history
                    try:
                        pdf_bytes = export_history_to_pdf(st.session_state.history)
                        download_link = get_download_link(pdf_bytes, "analysis_history.pdf", "Download PDF")
                        st.markdown(download_link, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating PDF: {str(e)}")
            else:
                st.warning("No data available to export. Please analyze some images first.")

    # Technical details section (expandable)
    with st.expander("Technical Details"):
        st.markdown("""
        ### Model Information
        
        **ResNet18**
        - Architecture: Residual Network with 18 layers
        - Pre-trained on ImageNet and fine-tuned for AI vs Real image detection
        - Input size: 224x224 pixels
        - Features: Residual connections that help with gradient flow during training
        
        **Vision Transformer (ViT)**
        - Architecture: Transformer-based model with self-attention mechanisms
        - Based on google/vit-base-patch16-224
        - Input size: 224x224 pixels
        - Features: Divides images into patches and processes them with transformer layers
        
        ### Processing Pipeline
        1. Image is resized to 224x224 pixels
        2. Pixel values are normalized using ImageNet mean and standard deviation
        3. Image is converted to tensor and passed through the model
        4. Softmax is applied to get class probabilities
        5. Results are interpreted and displayed
        
        ### Performance Considerations
        - First-time model loading may take a few seconds
        - GPU acceleration is used when available
        - Models are cached to improve subsequent analysis speed
        """)

    # Usage tips section
    with st.expander("Usage Tips"):
        st.markdown("""
        ### Getting the Best Results
        
        - Upload clear, well-lit images for more accurate detection
        - Try both models for comparison, as they may have different strengths
        - Higher confidence scores generally indicate more reliable predictions
        - The "Compare Both" option provides the most comprehensive analysis
        
        ### Understanding the Results
        
        - **Real**: The model predicts the image was created by a camera or human artist
        - **AI-generated**: The model predicts the image was created by AI tools like DALL-E, Midjourney, or Stable Diffusion
        - Confidence scores show how certain the model is about its prediction
        - When models disagree, consider the one with higher confidence
        
        ### Batch Processing
        
        - Use batch processing to analyze multiple images at once
        - Results are saved and can be exported to CSV or PDF
        - The gallery view provides a quick visual overview of results
        """)

    # Footer
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>AI vs Real Image Detection - Developed with Streamlit and PyTorch</p>", unsafe_allow_html=True)
