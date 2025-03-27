import pickle
from pathlib import Path

import joblib
import streamlit as st
import tensorflow as tf
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
        # Initialize the model if not provided
        if self.phobert is None:
            from transformers import TFAutoModel
            self.phobert = TFAutoModel.from_pretrained(self.phobert_name)
            self.phobert.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.phobert(input_ids=input_ids, attention_mask=attention_mask)[0]
        return output
        
    def get_config(self):
        config = super(CustomPhoBERTLayer, self).get_config()
        config.update({"phobert_name": self.phobert_name})
        return config
        
    @classmethod
    def from_config(cls, config):
        # Defensive programming to handle potentially None config
        if config is None:
            config = {}
        
        # Create a copy of config to avoid modifying the original
        config_copy = dict(config)
        
        # Safely extract phobert_name with a default if it doesn't exist
        phobert_name = config_copy.pop("phobert_name", "vinai/phobert-base")
        
        # Create and return the layer instance
        return cls(**config_copy)

@register_keras_serializable()
class CLSTokenExtractor(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CLSTokenExtractor, self).__init__(**kwargs)
        
    def call(self, inputs):
        return inputs[:, 0, :]
        
    def get_config(self):
        config = super(CLSTokenExtractor, self).get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        # Handle potential None config gracefully
        if config is None:
            config = {}
        return cls(**config)

def load_models():
    """List all available models in the model folder"""
    model_dir = Path("./model")
    if not model_dir.exists():
        st.error("Model directory not found. Please create a 'model' folder and add your models.")
        return {}
    
    models = {}
    for file in model_dir.glob("*.pkl"):
        models[file.stem] = str(file)
    
    for file in model_dir.glob("*.joblib"):
        models[file.stem] = str(file)
    
    # Add support for .h5 and .keras model files
    for file in model_dir.glob("*.h5"):
        models[file.stem] = str(file)
        
    for file in model_dir.glob("*.keras"):
        models[file.stem] = str(file)
        
    return models

def load_model(model_path):
    """Load a selected model"""
    try:
        if model_path.endswith('.pkl'):
            return pickle.load(open(model_path, 'rb'))
        elif model_path.endswith('.joblib'):
            return joblib.load(model_path)
        # Add support for loading Keras/TensorFlow models
        elif model_path.endswith('.h5') or model_path.endswith('.keras'):
            # Include custom objects when loading the model
            return tf.keras.models.load_model(model_path, custom_objects={
                'CustomPhoBERTLayer': CustomPhoBERTLayer,
                'CLSTokenExtractor': CLSTokenExtractor
            })
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def predict(model, text):
    """Make prediction using the loaded model"""
    try:
        # Check if model is a dictionary (some models store vectorizer separately)
        if isinstance(model, dict):
            vectorizer = model.get('vectorizer')
            classifier = model.get('model')
            
            if vectorizer and classifier:
                # Transform text using vectorizer
                text_vector = vectorizer.transform([text])
                # Make prediction
                prediction = classifier.predict(text_vector)[0]
                return prediction
        # Check if it's a Keras model
        elif isinstance(model, tf.keras.Model):
            # Check if the model has a CustomPhoBERTLayer (by inspecting its layers)
            has_phobert = any('CustomPhoBERTLayer' in str(layer.__class__) for layer in model.layers)
            
            if has_phobert:
                # Use the transformers tokenizer for PhoBERT
                from transformers import AutoTokenizer
                
                # Load the PhoBERT tokenizer
                tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
                
                # Tokenize the input text
                encoded = tokenizer(
                    text,
                    padding='max_length',
                    truncation=True,
                    max_length=256,  # Adjust as needed based on your model
                    return_tensors='tf'
                )
                
                # Extract input_ids and attention_mask
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
                
                # Make prediction with both inputs
                prediction = model.predict([input_ids, attention_mask])
                
                # For binary classification, threshold at 0.5
                return 1 if prediction[0][0] > 0.5 else 0
            else:
                # For other Keras models that don't use PhoBERT
                from tensorflow.keras.preprocessing.text import Tokenizer
                from tensorflow.keras.preprocessing.sequence import pad_sequences
                
                # Basic preprocessing
                tokenizer = Tokenizer()
                tokenizer.fit_on_texts([text])
                sequence = tokenizer.texts_to_sequences([text])
                padded = pad_sequences(sequence, maxlen=100)  # Adjust maxlen as needed
                
                prediction = model.predict(padded)
                # For binary classification, threshold at 0.5
                return 1 if prediction[0][0] > 0.5 else 0
        else:
            # Assume model has a predict method
            prediction = model.predict([text])[0]
            return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def main():
    st.title("Fake News Detection")
    st.markdown("This application helps you identify potential fake news articles.")
    
    # Load available models
    models = load_models()
    
    if not models:
        st.warning("No models found in the model directory.")
        return
    
    # Model selection
    selected_model_name = st.selectbox("Select a model", list(models.keys()))
    selected_model_path = models[selected_model_name]
    
    # Load the selected model
    with st.spinner("Loading model..."):
        model = load_model(selected_model_path)
    
    if model is None:
        st.error("Failed to load the model. Please select another one.")
        return
    
    # Text input
    st.subheader("Enter News Text")
    news_text = st.text_area("Paste the news article text here:", height=200)
    
    if st.button("Predict"):
        if not news_text:
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                # Make prediction
                result = predict(model, news_text)
                
                if result is not None:
                    # Display result with appropriate styling
                    if result == 1 or result == "FAKE":
                        st.error("📢 This appears to be FAKE NEWS!")
                    else:
                        st.success("✅ This appears to be REAL NEWS!")
                    
                    # Display confidence if available
                    if hasattr(model, 'predict_proba'):
                        try:
                            confidence = model.predict_proba([news_text])[0]
                            st.info(f"Confidence scores: {confidence}")
                        except:
                            pass

if __name__ == "__main__":
    main()
