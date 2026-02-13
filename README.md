🌾 **Crop Classification using Deep Learning**

📌 **Project Overview**

This project builds a deep learning-based image classification system capable of identifying **139 different crop types** using computer vision and transfer learning.

The model is trained on the Popular 139 Crops Image Dataset and deployed as an interactive web application using Streamlit.

The system enables automated crop recognition, supporting applications in:

Smart agriculture

Agricultural AI systems

Plant recognition tools

Precision farming

📂 **Dataset Information**

Dataset Name: Popular 139 Crops Image Dataset

📊 Total Classes: 139 crop types

🖼 Images per Class: ~250

📐 Image Size: 224 × 224 pixels

🎨 **Color Formats:**

RGB

BGR (OpenCV compatible)

Grayscale

📦 **Estimated Total Images:**

139 crops × 250 images × 3 formats ≈ 104,250 images

The dataset is uniformly resized and structured to support deep learning workflows.

🧠 **Model Architecture**

Base Model: ResNet18

Framework: PyTorch

Custom Fully Connected Layer for 139-class classification

Transfer Learning approach

Input size: 224×224

Image normalization applied

The final classification layer was modified to match the number of crop categories.

🔄 **Image Preprocessing**

The following transformations were applied:

Resize to 224×224

Convert to Tensor

Normalize using ImageNet mean & standard deviation

🚀 **Web Application (Streamlit)**

The model is deployed using Streamlit for real-time crop prediction.

Features:

Upload crop image (jpg, jpeg, png)

Real-time prediction

Displays predicted crop label

User-friendly interface

To run the app locally:

pip install -r requirements.txt
streamlit run crop_classifier.py

🛠 **Tech Stack**

Python

PyTorch

Torchvision

Streamlit

PIL

Joblib

Deep Learning (CNN)

📁 **Project Structure**
crop-classifier/
│
├── crop_classifier.py          # Streamlit app
├── crop_classifier_model.pkl   # Trained model
├── model_training.ipynb        # Model training notebook
├── README.md

📊 **Key Highlights**

✔ Multi-class classification (139 classes)
✔ Large-scale dataset (~100k+ images)
✔ Transfer learning implementation
✔ End-to-end pipeline (Training → Saving Model → Deployment)
✔ Interactive web application

🎯 **Future Improvements**

Add model confidence score

Deploy on Streamlit Cloud / HuggingFace Spaces

Add top-3 predictions

Improve accuracy with ResNet50 or EfficientNet

Add mobile compatibility
