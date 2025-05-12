import pandas as pd
import streamlit as st
import numpy as np
from streamlit_option_menu import option_menu
import time
import os
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import tensorflow as tf
from util import load_audio_from_folder
from util import train_decision_tree
from util import train_lstm
from util import train_ann
from util import predict_decision_tree_class
from util import predict_ann_class
from util import predict_lstm_class
from util import preprocess_and_extract_features_from_file
import tempfile

# Memuat data dari semua folder kategori
artifact_audio = load_audio_from_folder('Heartbeat_Sound/artifact', 'artifact')
extrahls_audio = load_audio_from_folder('Heartbeat_Sound/extrahls', 'extrahls')
extrastole_audio = load_audio_from_folder('Heartbeat_Sound/extrastole', 'extrastole')
murmur_audio = load_audio_from_folder('Heartbeat_Sound/murmur', 'murmur')
normal_audio = load_audio_from_folder('Heartbeat_Sound/normal', 'normal')

# Membuat DataFrame
df_artifact = pd.DataFrame(artifact_audio)
df_extrahls = pd.DataFrame(extrahls_audio)
df_extrastole = pd.DataFrame(extrastole_audio)
df_murmur = pd.DataFrame(murmur_audio)
df_normal = pd.DataFrame(normal_audio)

# Gabungkan semua DataFrame
df = pd.concat([df_artifact, df_murmur, df_normal, df_extrastole, df_extrahls], ignore_index=True)

with st.sidebar:
    selected = option_menu("Main Menu", ["Dashboard", 'Training Data', 'Testing'], 
        icons=['house', 'gear', 'book'], menu_icon="cast", default_index=0)
    selected

if selected == 'Dashboard':
    st.title("Dashboard :")
    st.subheader("Data Detak Jantung")

    # Menampilkan ringkasan
    st.write("Jumlah data per label:")
    st.write(df['label'].value_counts())

    # Filter berdasarkan label
    label_filter = st.multiselect("Filter berdasarkan label:", df['label'].unique(), default=df['label'].unique())
    filtered_df = df[df['label'].isin(label_filter)]

    # Tampilkan data
    st.dataframe(filtered_df[['filename', 'sampling_rate', 'duration', 'label']])

    # Plot distribusi jumlah data
    st.subheader("Distribusi Jumlah Data per Label")
    fig1, ax1 = plt.subplots()
    df['label'].value_counts().plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_ylabel("Jumlah File")
    ax1.set_xlabel("Label")
    ax1.set_title("Jumlah File Audio per Kategori")
    st.pyplot(fig1)

    # Distribusi durasi
    st.subheader("Distribusi Durasi Audio")
    fig2, ax2 = plt.subplots()
    df.boxplot(column='duration', by='label', ax=ax2)
    ax2.set_ylabel("Durasi (detik)")
    ax2.set_title("Distribusi Durasi Audio per Label")
    plt.suptitle("")
    st.pyplot(fig2)

    # Visualisasi waveform
    st.subheader("Visualisasi Waveform Audio")
    selected_file = st.selectbox("Pilih audio untuk ditampilkan waveform:", df['filename'])
    selected_row = df[df['filename'] == selected_file].iloc[0]

    signal = selected_row['signal']
    sr = selected_row['sampling_rate']
    time = np.linspace(0, len(signal) / sr, num=len(signal))

    fig3, ax3 = plt.subplots(figsize=(10, 3))
    ax3.plot(time, signal, color='purple')
    ax3.set_title(f"Waveform: {selected_row['label']}")
    ax3.set_xlabel("Waktu (detik)")
    ax3.set_ylabel("Amplitudo")
    st.pyplot(fig3)

    # Tambahkan audio player
    st.audio(selected_file, format='audio/wav')

elif selected == 'Training Data':
    st.title("Training Model :")
    
    # Filter based on label
    label_filter = st.multiselect("Filter based on label:", df['label'].unique(), default=df['label'].unique())
    filtered_df = df[df['label'].isin(label_filter)]

    # Display data
    st.dataframe(filtered_df[['filename', 'sampling_rate', 'duration', 'label']])

    # Tambahkan opsi model ANN
    model_option = st.selectbox("Select model:", ["Decision Tree", "LSTM", "ANN"])

    if st.button("Start Training"):
        with st.spinner("Training model is in progress..."):
            try:
                if model_option == "Decision Tree":
                    report = train_decision_tree(filtered_df)
                    st.success("Training completed!")
                    st.text(report)

                elif model_option == "LSTM":
                    model, history = train_lstm(filtered_df)
                    st.success("Training completed!")

                elif model_option == "ANN":
                    model, history = train_ann(filtered_df)  # Fungsi ini harus dari util.py
                    st.success("Training completed!")

            except Exception as e:
                st.error(f"An error occurred: {e}")

elif selected == 'Testing':
    st.title("Model Testing Interface")

    # Pilih model
    model_choice = st.selectbox("Pilih Model yang Ingin Digunakan:", ["Decision Tree", "ANN", "LSTM"])

    # Upload audio
    uploaded_file = st.file_uploader("Unggah file audio WAV", type=["wav"])

    if uploaded_file is not None:
        # Simpan file sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        if st.button("Lakukan Prediksi"):
            with st.spinner("Memproses dan melakukan prediksi..."):
                try:
                    if model_choice == "Decision Tree":
                        prediction = predict_decision_tree_class(tmp_file_path)
                    elif model_choice == "ANN":
                        prediction = predict_ann_class(tmp_file_path)
                    elif model_choice == "LSTM":
                        prediction = predict_lstm_class(tmp_file_path)
                    else:
                        prediction = None

                    if prediction is not None:
                        st.success(f"Hasil Prediksi ({model_choice}): {prediction}")
                    else:
                        st.error("Gagal melakukan prediksi. Pastikan file audio valid.")
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
