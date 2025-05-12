import os
import librosa
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import librosa
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
import matplotlib.pyplot as plt
import streamlit as st
import joblib
# from tensorflow.keras.models import load_model
from keras.initializers import Orthogonal
import pickle

def load_audio_from_folder(folder_path, label):
    audio_data = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'): 
            full_path = os.path.join(folder_path, filename)
            try:
                signal, sr = librosa.load(full_path, sr=None)  # sr=None untuk mempertahankan frekuensi asli
                duration = len(signal) / sr
                audio_data.append({
                    'filename': full_path,
                    'signal': signal,
                    'sampling_rate': sr,
                    'duration': duration,
                    'label': label
                })
            except Exception as e:
                print(f"Error reading {full_path}: {e}")

    return audio_data

# --- Signal Processing ---
def normalize_audio(signal):
    return signal / np.max(np.abs(signal))

def band_pass_filter(signal, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def clip_or_pad(signal, target_samples):
    if len(signal) < target_samples:
        signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')
    else:
        signal = signal[:target_samples]
    return signal

def preprocess(row, target_duration):
    signal = row['signal']
    sr = row['sampling_rate']
    target_samples = int(target_duration * sr)
    normalized_signal = normalize_audio(signal)
    processed_signal = clip_or_pad(normalized_signal, target_samples)
    filtered_signal = band_pass_filter(processed_signal, lowcut=100, highcut=1500, fs=sr)
    return filtered_signal

# --- Feature Extraction ---
def extract_features(signal, sr):
    intervals = np.diff(np.where(signal > 0)[0]) / sr
    mean_interval = np.mean(intervals) if len(intervals) > 0 else np.nan
    sdnn = np.std(intervals) if len(intervals) > 0 else np.nan
    rmssd = np.sqrt(np.mean(np.square(np.diff(intervals)))) if len(intervals) > 1 else np.nan

    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    fft_values = np.abs(np.fft.rfft(signal))
    peak_freq = freqs[np.argmax(fft_values)] if len(fft_values) > 0 else np.nan

    energy = np.sum(signal**2)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)

    zero_crossings = librosa.feature.zero_crossing_rate(signal)[0]
    zero_crossing_rate = np.mean(zero_crossings)

    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)

    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    rms = librosa.feature.rms(y=signal)[0]
    rms_mean = np.mean(rms)

    return {
        'mean_interval': mean_interval,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'peak_freq': peak_freq,
        'energy': energy,
        'mfcc_mean': mfcc_mean,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'chroma_mean': chroma_mean,
        'rms_mean': rms_mean
    }

def process_feature_extract(df, target_duration=6):
    df['processed_signal'] = df.apply(lambda row: preprocess(row, target_duration), axis=1)
    all_features = []
    for index, row in df.iterrows():
        signal = row['processed_signal']
        sr = row['sampling_rate']
        features = extract_features(signal, sr)
        features['filename'] = row['filename']
        all_features.append(features)
    return pd.DataFrame(all_features)

def train_decision_tree(df):
    features_df = process_feature_extract(df)

    mfcc_flattened = np.array(features_df['mfcc_mean'].tolist())
    chroma_flattened = np.array(features_df['chroma_mean'].tolist())

    X = np.hstack((
        features_df[['mean_interval', 'sdnn', 'rmssd', 'peak_freq', 'energy',
                     'zero_crossing_rate', 'spectral_centroid_mean',
                     'spectral_bandwidth_mean', 'rms_mean']].values,
        mfcc_flattened, chroma_flattened
    ))

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=42, stratify=y_train)

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    st.subheader("📊 Classification Report")
    st.code(report)

def normalize_audio_lstm(signal):
    """Normalizes the audio signal to the range [-1, 1]."""
    return signal / np.max(np.abs(signal))

def bad_pass_filter_lstm(signal, lowcut, highcut, fs, order=5):
    """Applies a band-pass filter to the audio signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def clip_or_pad_lstm(signal, target_samples):
    """Clips or pads the audio signal to the target length."""
    if len(signal) < target_samples:
        signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')
    else:
        signal = signal[:target_samples]
    return signal

def preprocess_lstm(row, target_duration):
    """Processes a single row of the DataFrame."""
    signal = row['signal']
    sr = row['sampling_rate']
    target_samples = int(target_duration * sr)
    
    normalized_signal = normalize_audio_lstm(signal)
    
    processed_signal = clip_or_pad_lstm(normalized_signal, target_samples)
    
    # Adjust the band-pass filter frequencies according to the characteristics of your data
    filtered_signal = bad_pass_filter_lstm(processed_signal, lowcut=100, highcut=1500, fs=sr)
    
    return filtered_signal

def extract_features_lstm(signal, sr):
    """Extracts multiple features from the audio signal."""
    intervals = np.diff(np.where(signal > 0)[0]) / sr
    mean_interval = np.mean(intervals) if len(intervals) > 0 else np.nan
    sdnn = np.std(intervals) if len(intervals) > 0 else np.nan
    rmssd = np.sqrt(np.mean(np.square(np.diff(intervals)))) if len(intervals) > 1 else np.nan
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    fft_values = np.abs(np.fft.rfft(signal))
    peak_freq = freqs[np.argmax(fft_values)] if len(fft_values) > 0 else np.nan
    energy = np.sum(signal**2)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    zero_crossings = librosa.feature.zero_crossing_rate(signal)[0]
    zero_crossing_rate = np.mean(zero_crossings)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    rms = librosa.feature.rms(y=signal)[0]
    rms_mean = np.mean(rms)

    return {
        'mean_interval': mean_interval,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'peak_freq': peak_freq,
        'energy': energy,
        'mfcc_mean': mfcc_mean,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'chroma_mean': chroma_mean,
        'rms_mean': rms_mean
    }

def preocess_feature_extract_lstm(df):
    """Extract features for all the audio signals in the dataframe."""
    all_features = []

    for index, row in df.iterrows():
        signal = row['processed_signal']
        sr = row['sampling_rate']
        features = extract_features_lstm(signal, sr)
        features['filename'] = row['filename']
        all_features.append(features)

    return pd.DataFrame(all_features)

def train_lstm(df):
    """Train LSTM model with the provided dataframe and display results in Streamlit."""
    target_duration = 6  # Target duration for padding/clipping
    df['processed_signal'] = df.apply(lambda row: preprocess_lstm(row, target_duration), axis=1)
    
    features_df = preocess_feature_extract_lstm(df)
    mfcc_flattened = np.array(features_df['mfcc_mean'].tolist())
    chroma_flattened = np.array(features_df['chroma_mean'].tolist())

    X = np.hstack((
        features_df[['mean_interval', 'sdnn', 'rmssd', 'peak_freq', 'energy',
                     'zero_crossing_rate', 'spectral_centroid_mean',
                     'spectral_bandwidth_mean', 'rms_mean']].values,
        mfcc_flattened, chroma_flattened
    ))

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['label'])

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=42, stratify=y_train)

    # One-hot encode labels
    num_classes = len(np.unique(y))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # Reshape for LSTM input
    x_train = np.expand_dims(x_train, axis=1)
    x_val = np.expand_dims(x_val, axis=1)
    x_test = np.expand_dims(x_test, axis=1)

    # Build LSTM model
    model = Sequential([
        LSTM(64, input_shape=(x_train.shape[1], x_train.shape[2]), return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train_cat, epochs=20, batch_size=32, validation_data=(x_val, y_val_cat), verbose=1)

    # Evaluation
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test_cat, axis=1)

    report = classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_, digits=2, output_dict=False)
    st.subheader("📊 Classification Report")
    st.code(report)

    # Visualize accuracy & loss
    st.subheader("📈 Training Performance")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax[0].plot(history.history['accuracy'], label='Train Accuracy')
    ax[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax[0].set_title('Model Accuracy')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[0].legend()
    ax[0].grid(True)

    # Loss plot
    ax[1].plot(history.history['loss'], label='Train Loss')
    ax[1].plot(history.history['val_loss'], label='Validation Loss')
    ax[1].set_title('Model Loss')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

    return model, history

def normalize_audio_ann(signal):
    """Normalizes the audio signal to the range [-1, 1]."""
    return signal / np.max(np.abs(signal))

def bad_pass_filter_ann(signal, lowcut, highcut, fs, order=5):
    """Applies a band-pass filter to the audio signal."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

def clip_or_pad_ann(signal, target_samples):
    """Clips or pads the audio signal to the target length."""
    if len(signal) < target_samples:
        signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')
    else:
        signal = signal[:target_samples]
    return signal

def preprocess_ann(row, target_duration):
    """Processes a single row of the DataFrame."""
    signal = row['signal']
    sr = row['sampling_rate']
    target_samples = int(target_duration * sr)
    
    normalized_signal = normalize_audio_ann(signal)
    
    processed_signal = clip_or_pad_ann(normalized_signal, target_samples)
    
    # Adjust the band-pass filter frequencies according to the characteristics of your data
    filtered_signal = bad_pass_filter_ann(processed_signal, lowcut=100, highcut=1500, fs=sr)
    
    return filtered_signal

def extract_features_ann(signal, sr):
    """Extracts multiple features from the audio signal."""
    intervals = np.diff(np.where(signal > 0)[0]) / sr
    mean_interval = np.mean(intervals) if len(intervals) > 0 else np.nan
    sdnn = np.std(intervals) if len(intervals) > 0 else np.nan
    rmssd = np.sqrt(np.mean(np.square(np.diff(intervals)))) if len(intervals) > 1 else np.nan
    freqs = np.fft.rfftfreq(len(signal), d=1/sr)
    fft_values = np.abs(np.fft.rfft(signal))
    peak_freq = freqs[np.argmax(fft_values)] if len(fft_values) > 0 else np.nan
    energy = np.sum(signal**2)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    zero_crossings = librosa.feature.zero_crossing_rate(signal)[0]
    zero_crossing_rate = np.mean(zero_crossings)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    chroma = librosa.feature.chroma_stft(y=signal, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    rms = librosa.feature.rms(y=signal)[0]
    rms_mean = np.mean(rms)

    return {
        'mean_interval': mean_interval,
        'sdnn': sdnn,
        'rmssd': rmssd,
        'peak_freq': peak_freq,
        'energy': energy,
        'mfcc_mean': mfcc_mean,
        'zero_crossing_rate': zero_crossing_rate,
        'spectral_centroid_mean': spectral_centroid_mean,
        'spectral_bandwidth_mean': spectral_bandwidth_mean,
        'chroma_mean': chroma_mean,
        'rms_mean': rms_mean
    }

def preprocess_extract_feature_ann(df):
    """Extract features for all the audio signals in the dataframe."""
    all_features = []

    for index, row in df.iterrows():
        signal = row['processed_signal']
        sr = row['sampling_rate']
        features = extract_features_ann(signal, sr)
        features['filename'] = row['filename']
        all_features.append(features)

    return pd.DataFrame(all_features)

def train_ann(df):
    """Train ANN model and visualize results in Streamlit."""

    # 1. Proses sinyal audio untuk semua baris
    target_duration = 3  # durasi target (detik)
    df['processed_signal'] = df.apply(lambda row: preprocess_ann(row, target_duration), axis=1)

    # 2. Ekstraksi fitur
    df_features = preprocess_extract_feature_ann(df)

    # 3. Gabungkan kembali label untuk training
    df_features = df_features.merge(df[['filename', 'label']], on='filename', how='left')

    # 4. Ekstraksi fitur numerik
    mfcc_flattened = np.array(df_features['mfcc_mean'].tolist())      # shape: (n_samples, 13)
    chroma_flattened = np.array(df_features['chroma_mean'].tolist())  # shape: (n_samples, 12)

    numerical_features = df_features[['mean_interval', 'sdnn', 'rmssd', 'peak_freq', 'energy',
                                      'zero_crossing_rate', 'spectral_centroid_mean',
                                      'spectral_bandwidth_mean', 'rms_mean']].values

    # 5. Gabungkan semua fitur menjadi X
    X = np.hstack((numerical_features, mfcc_flattened, chroma_flattened))

    # 6. Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_features['label'])

    # 7. Split dataset
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=42, stratify=y_train)

    # 8. One-hot encoding
    num_classes = len(np.unique(y))
    y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes)
    y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)

    # 9. Build ANN model
    model = Sequential([
        Dense(128, input_shape=(X.shape[1],), activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 10. Train model
    history = model.fit(x_train, y_train_cat,
                        epochs=20,
                        batch_size=32,
                        validation_data=(x_val, y_val_cat),
                        verbose=1)

    # 11. Visualisasi akurasi
    fig_acc, ax_acc = plt.subplots(figsize=(8, 5))
    ax_acc.plot(history.history['accuracy'], label='Train Accuracy')
    ax_acc.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax_acc.set_xlabel('Epochs')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_title('Model Accuracy')
    ax_acc.legend()
    ax_acc.grid(True)
    st.subheader("📈 Model Accuracy")
    st.pyplot(fig_acc)

    # 12. Visualisasi loss
    fig_loss, ax_loss = plt.subplots(figsize=(8, 5))
    ax_loss.plot(history.history['loss'], label='Train Loss')
    ax_loss.plot(history.history['val_loss'], label='Validation Loss')
    ax_loss.set_xlabel('Epochs')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_title('Model Loss')
    ax_loss.legend()
    ax_loss.grid(True)
    st.subheader("📉 Model Loss")
    st.pyplot(fig_loss)

    # 13. Evaluasi model
    y_pred_probs = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    y_true_classes = np.argmax(y_test_cat, axis=1)

    # 14. Classification report
    report = classification_report(y_true_classes, y_pred_classes, target_names=label_encoder.classes_, digits=2)
    st.subheader("📊 Classification Report")
    st.text(report)

    return model, history

def preprocess_and_extract_features_from_file(file_path, target_duration=6):
    try:
        signal, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None

    # Normalize
    normalized_signal = normalize_audio(signal)

    # Pad or clip
    target_samples = int(target_duration * sr)
    processed_signal = clip_or_pad(normalized_signal, target_samples)

    # Band-pass filter
    filtered_signal = band_pass_filter(processed_signal, lowcut=100, highcut=1500, fs=sr)

    # Extract features
    features = extract_features(filtered_signal, sr)

    # Flatten MFCC and Chroma
    mfcc_flat = features['mfcc_mean']
    chroma_flat = features['chroma_mean']

    # Combine features into a single vector
    feature_vector = np.hstack((
        np.array([
            features['mean_interval'], features['sdnn'], features['rmssd'],
            features['peak_freq'], features['energy'],
            features['zero_crossing_rate'], features['spectral_centroid_mean'],
            features['spectral_bandwidth_mean'], features['rms_mean']
        ]),
        mfcc_flat,
        chroma_flat
    ))

    return feature_vector.reshape(1, -1)  # Reshape for model prediction

def predict_decision_tree_class(file_path):
    clf = joblib.load('model/decision_tree_model.pkl')
    label_encoder = joblib.load('model/label_encoder.pkl')
    features = preprocess_and_extract_features_from_file(file_path)
    if features is None:
        return None
    prediction_class = clf.predict(features)[0]
    return label_encoder.inverse_transform([prediction_class])[0]

def predict_ann_class(file_path):
    model = load_model('model/ann_model.h5')
    label_encoder = joblib.load('model/ann_label_encoder.pkl')
    features = preprocess_and_extract_features_from_file(file_path)
    if features is None:
        return None
    probs = model.predict(features)
    prediction_class = np.argmax(probs, axis=1)[0]
    return label_encoder.inverse_transform([prediction_class])[0]

def predict_lstm_class(file_path):
    model = load_model('model/lstm_model.h5')
    label_encoder = joblib.load('model/lstm_label_encoder.pkl')
    features = preprocess_and_extract_features_from_file(file_path)
    if features is None:
        return None
    probs = model.predict(features)
    prediction_class = np.argmax(probs, axis=1)[0]
    return label_encoder.inverse_transform([prediction_class])[0]