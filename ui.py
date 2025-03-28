import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
import tensorflow as tf
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


def evaluate_model(model, X_test, y_test):
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

    models = {"dummy_model": None}  # Replace with actual model loading logic
    selected_model_name = st.selectbox("Select a model", list(models.keys()))
    model = models[selected_model_name]

    if model:
        st.subheader("Enter News Text")
        news_text = st.text_area("Paste the news article text here:", height=200)

        if st.button("Predict"):
            result = model.predict([news_text])[0]
            st.success("FAKE NEWS" if result == 1 else "REAL NEWS")

        if st.button("Evaluate Model"):
            X_test = np.random.rand(100, 256)  # Replace with actual test data
            y_test = np.random.randint(0, 2, 100)  # Replace with actual labels
            evaluate_model(model, X_test, y_test)
    else:
        st.warning("No models found.")


if __name__ == "__main__":
    main()
