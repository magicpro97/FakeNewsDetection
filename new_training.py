from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
import re
import string
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'\d+', '', text)  # Loại bỏ số
    text = text.translate(str.maketrans(
        '', '', string.punctuation))  # Loại bỏ dấu câu
    text = re.sub(r'\s+', ' ', text).strip()  # Loại bỏ khoảng trắng thừa
    return text


df["Content"] = df["Content"].apply(clean_text)

# Chia dữ liệu train/test
X_train, X_test, y_train, y_test = train_test_split(
    df["Content"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Tokenization với LSTM (Deep Learning)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_length = 200  # Giới hạn độ dài của mỗi văn bản
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding="post")

# Deep Learning với LSTM

model = Sequential([
    Embedding(input_dim=10000, output_dim=128, input_length=max_length),
    LSTM(64, return_sequences=True),
    LSTM(32),
    Dropout(0.5),
    Dense(1, activation="sigmoid")
])

model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])

# Huấn luyện mô hình
model.fit(X_train_pad, y_train, epochs=5, batch_size=32,
          validation_data=(X_test_pad, y_test))

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
news_real = f"Thủ tướng Phạm Minh Chính cho biết Việt Nam luôn coi trọng hợp tác với EU và đề nghị EU sớm hoàn tất phê chuẩn Hiệp định Bảo hộ đầu tư Việt Nam-EU."
news_fake = f"Nhà khoa học Việt Nam chế tạo thành công máy phát điện vĩnh cửu không cần nhiên liệu, có thể cung cấp điện miễn phí cho toàn bộ đất nước."

print(predict_fake_news(news_real))
print(predict_fake_news(news_fake))
