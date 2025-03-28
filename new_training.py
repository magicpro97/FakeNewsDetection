from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import requests


# Tải danh sách stopwords từ VietAI GitHub
url = "https://raw.githubusercontent.com/stopwords/vietnamese-stopwords/master/vietnamese-stopwords.txt"
stop_words_vietnamese = set(requests.get(url).text.split("\n"))

# Đọc dữ liệu tin thật
df_real = pd.read_csv("./data/vnexpress_dataset.csv")
df_real["label"] = 0  # 0: Tin thật

# Đọc dữ liệu tin giả
df_fake = pd.read_csv("./data/vnexpress_fake_dataset.csv")
df_fake["label"] = 1  # 1: Tin giả

# Gộp hai bộ dữ liệu
df = pd.concat([df_real, df_fake], ignore_index=True)

# Xử lý nội dung trống
df = df.dropna(subset=["Content"])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words_vietnamese]
    return " ".join(words)


df["Content"] = df["Content"].apply(clean_text)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["Content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Tokenization với LSTM (Deep Learning)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 500  # Giới hạn độ dài của mỗi văn bản
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding="post")

# Deep Learning với LSTM
model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),  # Bi-LSTM
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(16, activation="relu"),  # Thêm một hidden layer để học tốt hơn
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(X_train_pad, y_train, epochs=5, batch_size=32,
          validation_data=(X_test_pad, y_test))


# Lưu mô hình dưới dạng file .keras
model.save("./model/fake_news_model_LSMT.keras")

# Kiểm tra mô hình
y_pred_prob = model.predict(X_test_pad)
y_pred = (y_pred_prob > 0.5).astype(int)
print(classification_report(y_test, y_pred))


# Dự đoán tin giả
def predict_fake_news(text):
    text = clean_text(text)
    text_seq = tokenizer.texts_to_sequences([text])
    text_pad = pad_sequences(text_seq, maxlen=max_length, padding="post")

    prob = model.predict(text_pad)[0][0]  # Xác suất tin giả

    label = "🛑 Tin giả" if prob > 0.5 else "✅ Tin thật"
    confidence = f"{prob*100:.2f}%" if prob > 0.5 else f"{(1-prob)*100:.2f}%"

    return f"{label} (Confidence: {confidence})"


# Ví dụ kiểm tra tin tức
news_real = f"Công ty của diễn viên hài Thu Trang phải trả 1,3 tỷ đồng cho đối tác"
news_fake = f"Công ty của diễn viên hài Huy Pham phải trả 15 tỷ đồng cho đối tác"

print(predict_fake_news(news_real))
print(predict_fake_news(news_fake))
