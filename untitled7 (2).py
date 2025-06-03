import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

st.title("Analisis Sentimen Ulasan Produk")

# Load dataset
url = "https://raw.githubusercontent.com/Jujun8/sansan/main/data%20proyek.csv"
df = pd.read_csv(url)
df.columns = df.columns.str.strip().str.lower()

# Cek nama kolom
st.write("Kolom tersedia:", df.columns.tolist())

# Pastikan kolom
text_column = 'content'
target_column = 'score'

# Validasi kolom
if text_column not in df.columns or target_column not in df.columns:
    st.error(f"Kolom '{text_column}' atau '{target_column}' tidak ditemukan dalam dataset.")
    st.stop()

# Bersihkan data
df.dropna(subset=[text_column, target_column], inplace=True)
df[text_column] = df[text_column].astype(str).str.lower()

# Tampilkan data awal
st.subheader("Contoh Data")
st.dataframe(df.head())

# Split data
X = df[text_column]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluasi
y_pred = model.predict(X_test_tfidf)
st.subheader("Laporan Klasifikasi")
st.text(classification_report(y_test, y_pred))

# Prediksi ulasan baru
ulasan_baru = ["Barangnya bagus dan sesuai deskripsi."]
ulasan_baru2 = ["Produknya mengecewakan, tidak seperti yang diharapkan."]

pred1 = model.predict(tfidf.transform(ulasan_baru))[0]
pred2 = model.predict(tfidf.transform(ulasan_baru2))[0]

st.subheader("Prediksi Ulasan Baru")
st.write(f"Ulasan 1: '{ulasan_baru[0]}' → Skor: **{pred1}**")
st.write(f"Ulasan 2: '{ulasan_baru2[0]}' → Skor: **{pred2}**")
