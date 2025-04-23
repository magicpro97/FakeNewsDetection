import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from tf_keras.src.saving.object_registration import register_keras_serializable
from transformers import AutoTokenizer
from st_aggrid import AgGrid, GridOptionsBuilder, DataReturnMode, GridUpdateMode
from vnexpress_crawler import main as crawl_main

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
# reuse your clean_text function from training:
import re, string, requests

# (re)define your clean_text exactly as in training:
stop_words_url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
stop_words_vietnamese = set(requests.get(stop_words_url).text.split("\n"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [w for w in words if w not in stop_words_vietnamese]
    return " ".join(words)

# Set page configuration
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# Define custom layers for model loading
@register_keras_serializable()
class CustomPhoBERTLayer(tf.keras.layers.Layer):
    def __init__(self,
                 phobert_name="vinai/phobert-base",
                 phobert_model=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.phobert_name = phobert_name
        if phobert_model is None:
            from transformers import TFAutoModel
            self.phobert = TFAutoModel.from_pretrained(self.phobert_name)
            self.phobert.trainable = False
        else:
            self.phobert = phobert_model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        # returns last_hidden_state
        return self.phobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )[0]

    def get_config(self):
        config = super().get_config()
        # inject your custom arg
        config.update({
            "phobert_name": self.phobert_name,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # extract your arg before passing the rest
        return cls(**config)


@register_keras_serializable()
class CLSTokenExtractor(tf.keras.layers.Layer):
    def call(self, inputs):
        return inputs[:, 0, :]


def load_models():
    """Load available models from the ./data folder"""
    model_dir = Path("./model")
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

def evaluate_phobert(model, tokenizer, texts, labels, max_length=256, batch_size=32):
    """
    Tokenizes `texts` with `tokenizer`, runs `model.predict` in batches,
    then computes and displays classification metrics in Streamlit.
    
    Args:
      model: A loaded TFAutoModelForSequenceClassification (sigmoid head).
      tokenizer: The matching AutoTokenizer.
      texts:   List of raw string articles.
      labels:  List or array of 0/1 ground‐truth labels.
      max_length: Tokenization/truncation length.
      batch_size:  Inference batch size.
    """
    # 1) Tokenize all texts
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="tf"
    )
    input_ids      = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    
    # 2) Predict in batches
    preds = []
    n = input_ids.shape[0]
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_ids   = input_ids[start:end]
        batch_mask  = attention_mask[start:end]
        logits      = model.predict(
            {"input_ids": batch_ids, "attention_mask": batch_mask},
            verbose=0
        )
        # if your head returns logits, apply sigmoid:
        if logits.shape[-1] == 1:
            probs = tf.sigmoid(logits).numpy().flatten()
        else:
            # e.g. if shape=(batch,2) with softmax, take class-1:
            probs = tf.nn.softmax(logits, axis=-1).numpy()[:,1]
        preds.extend(probs)
    
    preds = np.array(preds)
    y_pred_classes = (preds > 0.5).astype(int)
    
    # 3) Compute metrics
    report     = classification_report(labels, y_pred_classes, output_dict=True)
    conf_matrix= confusion_matrix(labels, y_pred_classes)
    roc_auc    = roc_auc_score(labels, preds)
    avg_prec   = average_precision_score(labels, preds)
    
    # 4) Display in Streamlit
    st.subheader("📊 Evaluation Metrics")
    st.json(report)
    
    st.subheader("🗂️ Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    
    st.subheader("📈 ROC AUC Score")
    st.write(f"ROC AUC: **{roc_auc:.4f}**")
    
    st.subheader("🎯 Average Precision Score")
    st.write(f"Average Precision: **{avg_prec:.4f}**")

def main():
    st.title("Vietnamese Fake News Detector")
    st.markdown("This application helps you identify potential Vietnamese fake news articles.")

    tab1, tab2, tab3 = st.tabs(["Prediction", "Dataset Visualization", "Data Crawling"])

    with tab1:
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
                    if selected_model_name == "fake_news_model_LSMT":
                        with open("./data/lstm_tokenizer.pkl", "rb") as f:
                            lstm_tokenizer = pickle.load(f)
                            cleaned = clean_text(news_text)
                            seq = lstm_tokenizer.texts_to_sequences([cleaned])
                            padded = pad_sequences(seq,
                                                maxlen=500,           # your LSTM max_length
                                                padding="post",
                                                truncating="post")
                            probs = model.predict(padded)
                            prob = float(probs.squeeze())

                            if prob > 0.5:
                                st.error(f"📢 FAKE NEWS (p={prob:.2f})")
                            else:
                                st.success(f"✅ REAL NEWS (p={prob:.2f})")
                    else:
                        # Tokenize 
                        inputs = tokenizer(
                            news_text,
                            return_tensors="tf",
                            padding="max_length",
                            truncation=True,
                            max_length=256,
                        )

                        # Predict
                        # your model takes a dict of tensors, not a raw string
                        probs = model.predict({
                            "input_ids":    inputs["input_ids"],
                            "attention_mask": inputs["attention_mask"],
                        })
                        # if your final layer is Dense(1, activation="sigmoid")
                        prob = tf.squeeze(probs).numpy()

                        # Display
                        if prob > 0.5:
                            st.error(f"📢 FAKE NEWS (p={prob:.2f})")
                        else:
                            st.success(f"✅ REAL NEWS (p={prob:.2f})")

        if st.button("Evaluate Model"):
            if selected_model_name == "fake_news_model_LSMT":
                X_test = np.random.rand(100, 256)  # Replace with actual test data
                y_test = np.random.randint(0, 2, 100)  # Replace with actual labels
                evaluate_model(model, X_test, y_test)
            else:
                # 1. Load the combined CSV (adjust path as needed)
                df = pd.read_csv("./data/vnexpress_combined_dataset.csv")

                # 2. Convert Label to int (0=real, 1=fake)
                df["Label"] = df["Label"].fillna(0).astype(float)
                df = df.dropna(subset=["Content"])
                df["Content"] = df["Content"].astype(str)

                # 3. Extract the raw texts and labels
                texts = df["Content"].tolist()
                labels = df["Label"].tolist()

                from sklearn.model_selection import train_test_split

                # 4. Split off a test set (e.g. 20% of the data)
                _, X_test_texts, _, y_test_labels = train_test_split(
                    texts,
                    labels,
                    test_size=0.2,
                    random_state=42,
                    stratify=labels
                )
                evaluate_phobert(
                    model=model,
                    tokenizer=tokenizer,
                    texts=X_test_texts,
                    labels=y_test_labels,
                    max_length=256,
                    batch_size=32
                )

    with tab2:
        st.title("Real vs. Generated Fake News")
        st.markdown("This section allows you to visualize the dataset, viewing real news and generated fake news.")

        real_df = pd.read_csv("./data/vnexpress_dataset.csv",engine="python",header=0,usecols=[0, 1, 2, 3, 4])
        fake_df = pd.read_csv("./data/vnexpress_fake_dataset_enhance.csv",engine="python",header=0,usecols=[0, 1, 2, 3, 4])

        # 2) Build AgGrid options for single‐row selection
        gb = GridOptionsBuilder.from_dataframe(real_df)
        gb.configure_selection("single")  
        grid_opts = gb.build()

        # 3) Render the grid
        grid_response = AgGrid(
            real_df,
            gridOptions=grid_opts,
            height=300,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED, 
            update_mode=GridUpdateMode.SELECTION_CHANGED,
        )

        # 4) Grab the selected row
        rows = grid_response["selected_rows"]
        if type(rows) == pd.core.frame.DataFrame:
            records = rows.to_dict(orient="records")
            st.markdown("**You clicked on:**")
            st.json(records, expanded=False)

            # 5) Show fakes matching this real id
            sel_id = records[0]["Link"]
            st.subheader("Generated Fakes")
            fakes = fake_df[fake_df["Link"] == sel_id]
            if fakes.empty:
                st.write("No generated fakes for this article.")
            else:
                st.dataframe(fakes)
        else:
            st.write("Click on a real‑news row above to see its generated fakes.")

    with tab3:
        st.title("Data Crawling")
        st.markdown("This section allows you to crawl data from VNExpress.")
        if st.button("Crawl Data"):
            with st.spinner("Crawling..."):
                result = crawl_main()
                st.write(f"Crawled {len(result['all_data'])} articles with a total of {len(result['categories'])} categories in {result['total_time']:.2f} seconds, and saved to {result['CSV_FILE']}")
                st.write("Articles Crawled:")
                st.dataframe(result['all_data'])

if __name__ == "__main__":
    main()