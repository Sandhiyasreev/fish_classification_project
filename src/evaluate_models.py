from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model_path, data_gen):
    model = load_model(model_path)
    preds = model.predict(data_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = data_gen.classes
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=list(data_gen.class_indices.keys())))
