import os
import pickle
import joblib
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from tf_keras.src.saving.object_registration import register_keras_serializable


# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)


# Define custom layers for model loading
@register_keras_serializable()
class CustomPhoBERTLayer(tf.keras.layers.Layer):
    def __init__(self, phobert_model=None, **kwargs):
        super(CustomPhoBERTLayer, self).__init__(**kwargs)
        self.phobert = phobert_model
        self.phobert_name = "vinai/phobert-base"
        if self.phobert is None:
            from transformers import TFAutoModel
            self.phobert = TFAutoModel.from_pretrained(self.phobert_name)
            self.phobert.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output


@register_keras_serializable()
class CLSTokenExtractor(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, 0, :]


def load_models():
    """Load available models from the ./data folder"""
    model_dir = Path("./data")
    models = {}

    if not model_dir.exists():
        os.makedirs(model_dir)  # Ensure the directory exists
        return models

    for file in model_dir.glob("*.pkl"):
        models[file.stem] = str(file)
    for file in model_dir.glob("*.joblib"):
        models[file.stem] = str(file)
    for file in model_dir.glob("*.h5"):
        models[file.stem] = str(file)
    for file in model_dir.glob("*.keras"):
        models[file.stem] = str(file)

    return models


def load_model(model_path):
    """Load a selected model based on file extension"""
    try:
        if model_path.endswith('.pkl'):
            return pickle.load(open(model_path, 'rb'))
        elif model_path.endswith('.joblib'):
            return joblib.load(model_path)
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            return tf.keras.models.load_model(model_path, custom_objects={
                'CustomPhoBERTLayer': CustomPhoBERTLayer,
                'CLSTokenExtractor': CLSTokenExtractor
            })
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


def evaluate_model(model, X_test, y_test):
    """Evaluate the model using classification metrics"""
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    report = classification_report(y_test, y_pred_classes, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    roc_auc = roc_auc_score(y_test, y_pred)
    avg_precision = average_precision_score(y_test, y_pred)

    st.subheader("Evaluation Metrics")
    st.json(report)

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.subheader("ROC AUC Score")
    st.write(f"ROC AUC: {roc_auc:.4f}")

    st.subheader("Average Precision Score")
    st.write(f"Average Precision: {avg_precision:.4f}")


def main():
    st.title("Fake News Detection")
    st.markdown("This application helps you identify potential fake news articles.")

    models = load_models()

    if not models:
        st.warning("No models found in the ./data folder.")
        return

    selected_model_name = st.selectbox("Select a model", list(models.keys()))
    selected_model_path = models[selected_model_name]

    with st.spinner("Loading model..."):
        model = load_model(selected_model_path)

    if model is None:
        st.error("Failed to load the model.")
        return

    st.subheader("Enter News Text")
    news_text = st.text_area("Paste the news article text here:", height=200)

    if st.button("Predict"):
        if not news_text:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                prediction = model.predict([news_text])[0]
                if prediction == 1:
                    st.error("📢 This appears to be FAKE NEWS!")
                else:
                    st.success("✅ This appears to be REAL NEWS!")

    if st.button("Evaluate Model"):
        X_test = np.random.rand(100, 256)  # Replace with actual test data
        y_test = np.random.randint(0, 2, 100)  # Replace with actual labels
        evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()