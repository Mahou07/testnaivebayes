
import streamlit as st
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os
import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

if os.path.exists('./dataset.xlsx'):
    df = pd.read_excel('dataset.xlsx', index_col=None)

with st.sidebar:
    st.image("NaiveBayes.webp")
    st.title("Machinelearning")
    choice = st.radio(
        "Navigation", ["Upload", "Profiling", "Predict"])
    st.info("This project application helps you build and explore your data.")

if choice == "Upload":

    st.title("Upload Your Dataset")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:

        st.image("Upload.webp")

    with col3:
        st.write(' ')

    file = st.file_uploader("Upload Your Dataset")
    if file:
        df = pd.read_excel(file, index_col=None)
        df.to_excel('dataset.xlsx', index=None)
        st.dataframe(df)

if choice == "Profiling":
    st.title("Exploratory Data Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(' ')

    with col2:

        st.image("Profiling.webp")

    with col3:
        st.write(' ')

    profile_df = df.profile_report()
    st_profile_report(profile_df)


if choice == "Predict":
    def prepare_data(df):
        features = ["nilai_mtk", "nilai_bing", "minat",
                    "kemampuan", "kepribadian", "penghasilan_orangtua"]
        X = df[features]
        y = df["jurusan"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test

    # Fungsi untuk melatih model
    def train_model(X_train, y_train):
        model = GaussianNB()
        model.fit(X_train, y_train)
        return model

    # Fungsi untuk memprediksi jurusan siswa
    def predict(model, X_test):
        y_pred = model.predict(X_test)
        return y_pred

    # Mempersiapkan data untuk pemodelan
    X_train, X_test, y_train, y_test = prepare_data(df)

    # Melatih model
    model = train_model(X_train, y_train)

    st.title("Aplikasi Prediksi Jurusan Siswa")
    st.write("Isi data berikut untuk memprediksi jurusan siswa.")

    # Input form untuk data siswa
    nilai_mtk = st.slider("Nilai Matematika", 0, 100, 50)
    nilai_bing = st.slider("Nilai Bahasa Inggris", 0, 100, 50)
    minat = st.slider("Minat", 1, 5, 3)
    kemampuan = st.slider("Kemampuan", 1, 5, 3)
    kepribadian = st.slider("Kepribadian", 1, 5, 3)
    penghasilan_orangtua = st.slider(
        "Penghasilan Orang Tua", 0, 100000000, 5000000)

    # Menampilkan input data
    st.write("Data yang dimasukkan:")
    st.write("Nilai Matematika:", nilai_mtk)
    st.write("Nilai Bahasa Inggris:", nilai_bing)
    st.write("Minat:", minat)
    st.write("Kemampuan:", kemampuan)
    st.write("Kepribadian:", kepribadian)
    st.write("Penghasilan Orang Tua:", penghasilan_orangtua)

    # Prediksi jurusan siswa berdasarkan input data
    prediction_data = [[nilai_mtk, nilai_bing, minat,
                        kemampuan, kepribadian, penghasilan_orangtua]]
    y_pred = predict(model, prediction_data)
    st.write("Jurusan yang diprediksi:", y_pred[0])

    # Menampilkan hasil prediksi
