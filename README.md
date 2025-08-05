# 🐟 Fish Image Classification: CNN & Transfer Learning 🎯📷🧠

A deep learning project that classifies multiple fish species using Convolutional Neural Networks (CNN) and Transfer Learning techniques. Built to assist fisheries, marine biologists, and hobbyists in identifying fish from images accurately and efficiently.

---

# 🔧 Features

🔍 **Multiclass Image Classification**

📷 Trained on a dataset of labeled fish species  
🎓 Two model options:
- **Custom CNN model**
- **Transfer Learning** (using MobileNetV2 or VGG16)

📈 **Evaluation Metrics**
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- Classification Report

🧪 **Interactive Streamlit App**
- Upload fish image and predict species
- Choose between custom CNN or pre-trained model
- View metrics and prediction report
- Download classification results as PDF

---

# 🧠 Technologies Used

- **Python** (TensorFlow, Keras, OpenCV, NumPy, Pandas)
- **Convolutional Neural Networks**
- **Transfer Learning** (MobileNetV2, VGG16)
- **Matplotlib / Seaborn** (visualization)
- **Streamlit** (deployment and interactive UI)
- **Sklearn** (model evaluation)
- **ReportLab** (for PDF report generation)

---

# 📁 Project Structure

fish_classification_project/
├── app.py                            # Streamlit web app  
├── src/
│   ├── model_cnn.py                  # Custom CNN model architecture
│   ├── model_transfer.py             # Transfer learning models
│   ├── utils.py                      # Prediction, preprocessing, evaluation
├── models/
│   ├── cnn_model.h5                  # Trained CNN model
│   ├── mobilenet_model.h5           # MobileNetV2 model
│   ├── vgg16_model.h5               # VGG16 model
├── data/
│   ├── fish_images/                 # Raw fish image dataset
│   └── labels.csv                   # Image labels
├── notebooks/
│   └── Fish_Classification_Notebook.ipynb  # Model development notebook
├── outputs/
│   ├── confusion_matrix.png         # Evaluation visual
│   └── classification_report.pdf    # Generated prediction reports
├── requirements.txt                 # Required Python packages  
└── README.md                        # Project documentation

---

# 🧪 Example Use Cases

🎣 Marine Biologists → Automatically classify fish species in underwater imagery  
🛒 E-commerce → Auto-tag fish species in product listings  
📱 Mobile Apps → Build a "Fish Identifier" for educational or fishing apps  
🏢 Aquaculture Industry → Track and classify fish species for management  

---

## 🙋‍♀️ Created By

**Sandhiya Sree V**  
📧 sandhiyasreev@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/sandhiya-sree-v-3a2321298/)  
🌐 [GitHub](https://github.com/Sandhiyasreev)

---

# 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and share with proper credit.

⭐ **If you found this project helpful, please consider giving it a star!**  
💬 For feedback or collaboration, feel free to reach out!
