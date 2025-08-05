import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from tensorflow.keras.models import load_model

@st.cache_resource
def load_fish_model():
    return load_model("models/mobilenet_model.h5")

model = load_fish_model()

# =================== Configuration ===================
MODEL_PATHS = {
    "CNN": "models/mobilenet_model.h5",
    "ResNet50": "models/resnet50_model.h5",
    "VGG16": "models/vgg16_model.h5"
}
CLASS_NAMES = ['Black Sea Sprat', 'Gilt-Head Bream', 'Hourse Mackerel',
               'Red Mullet', 'Red Sea Bream', 'Sea Bass', 'Shrimp', 'Striped Red Mullet', 'Trout']  # replace with actual

# =================== Streamlit UI ===================
st.set_page_config(page_title="Fish Image Classifier", layout="centered")
st.title("🐟 Fish Image Classifier")
st.write("Upload an image of a fish to classify it using a deep learning model.")

# --------- Model Selection Dropdown ---------
model_choice = st.selectbox("Select Model to Use", list(MODEL_PATHS.keys()))
model_path = MODEL_PATHS[model_choice]
model = load_model(model_path)
st.write(f"🧠 **Model Used:** {model_choice}")

# --------- File Upload ---------
uploaded_file = st.file_uploader("📷 Upload a fish image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # --------- Prediction ---------
    prediction = model.predict(img_array)
    pred_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    st.success(f"🎯 **Predicted:** {pred_class}")
    st.info(f"🔍 **Confidence:** {confidence:.2f}")

    # --------- Evaluation Metrics Section ---------
    with st.expander("📊 View Model Evaluation Metrics"):
        try:
            # Dummy example metrics (replace with your actual test set eval)
            st.write("Accuracy: 94.7%")
            st.write("Precision: 95.1%")
            st.write("Recall: 93.8%")
            st.write("F1-score: 94.4%")
        except Exception as e:
            st.warning(f"Metrics not available: {e}")

    # --------- Confusion Matrix Button ---------
    if st.button("🧮 Show Confusion Matrix"):
        try:
            # Simulated test data prediction (for demo, use real y_test and y_pred)
            y_true = [0, 1, 2, 1, 0, 2, 1, 2, 0, 1]  # dummy data
            y_pred = [0, 1, 2, 0, 0, 2, 1, 2, 2, 1]
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASS_NAMES[:len(set(y_true))], yticklabels=CLASS_NAMES[:len(set(y_true))])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.title("Confusion Matrix")
            st.pyplot(plt)
        except Exception as e:
            st.error(f"Error displaying confusion matrix: {e}")

    # --------- Download Prediction Report ---------
    report = {
        "predicted_class": pred_class,
        "confidence": round(confidence, 2),
        "model_used": model_choice,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    report_json = json.dumps(report, indent=4)
    st.download_button("📥 Download Prediction Report", report_json, file_name="prediction_report.json", mime="application/json")
