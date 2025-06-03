# Install dependencies (jika belum)
# !pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load dataset
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df = pd.read_csv(url)
print(df.columns.tolist())


# Cek nama kolom
print("Kolom tersedia:", df.columns)

# Pastikan kolom 'ulasan' dan 'skor' sesuai dengan dataset Anda
text_column = 'content'
target_column = 'score'


# Validasi kolom
if text_column not in df.columns or target_column not in df.columns:
    raise ValueError(f"Kolom '{text_column}' atau '{target_column}' tidak ditemukan dalam dataset.")

# Hapus baris kosong
df.dropna(subset=[text_column, target_column], inplace=True)

# Preprocessing
df[text_column] = df[text_column].astype(str).str.lower()

# Pisahkan fitur dan target
X = df[text_column]
y = df[target_column]

# Split data training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluasi model
y_pred = model.predict(X_test_tfidf)
print("Laporan klasifikasi:")
print(classification_report(y_test, y_pred))

# Prediksi ulasan baru
ulasan_baru = ["Barangnya bagus dan sesuai deskripsi."]
ulasan_baru_tfidf = tfidf.transform(ulasan_baru)
prediksi = model.predict(ulasan_baru_tfidf)
print(f"Prediksi skor ulasan: {prediksi[0]}")

ulasan_baru2 = ["Produknya mengecewakan, tidak seperti yang diharapkan."]
ulasan_baru2_tfidf = tfidf.transform(ulasan_baru2)
prediksi2 = model.predict(ulasan_baru2_tfidf)
print(f"Prediksi skor ulasan: {prediksi2[0]}")
