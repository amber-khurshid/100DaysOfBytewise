import streamlit as st

import numpy as np
import librosa
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


# # Load the pre-trained model
# model_path = 'C:/Users/HP/OneDrive/Desktop/100DaysOfBytewise/project 2/rfc.pkl'  # Update with the path to your .pkl model file
# model = joblib.load(model_path)

# label_mapping= {0: 'blues', 1: 'classical', 2: 'country',3: 'disco',4: 'hiphop', 5: 'jazz',6: 'metal',7: 'pop',8: 'reggae',9: 'rock'}

# # Function to extract features from the audio file
# def extract_features(audio_path):
#     # Load the audio file
#     y, sr = librosa.load(audio_path, sr=None)
    
#     # Extract features


#     features = {
#         'length': len(y),
#         'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
#         'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
#         'rms_mean': np.mean(librosa.feature.rms(y=y)),
#         'rms_var': np.var(librosa.feature.rms(y=y)),
#         'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
#         'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
#         'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
#         'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=y)),
#         'harmony_mean': np.mean(librosa.feature.tempogram(y=y, sr=sr)),
#         'perceptr_mean': np.mean(librosa.feature.spectral_flatness(y=y)),
#         'perceptr_var': np.var(librosa.feature.spectral_flatness(y=y)),
#         'tempo': np.mean(librosa.beat.tempo(y=y, sr=sr)),
#         'mfcc1_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[0]),
#         'mfcc1_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[0]),
#         'mfcc2_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1]),
#         'mfcc3_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[2]),
#         'mfcc3_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[2]),
#         'mfcc4_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[3]),
#         'mfcc4_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[3]),
#         'mfcc5_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[4]),
#         'mfcc5_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[4]),
#         'mfcc6_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[5]),
#         'mfcc6_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[5]),
#         'mfcc7_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[6]),
#         'mfcc7_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[6]),
#         'mfcc8_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[7]),
#         'mfcc8_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[7]),
#         'mfcc9_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[8]),
#         'mfcc9_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[8]),
#         'mfcc10_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[9]),
#         'mfcc10_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[9]),
#         'mfcc11_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[10]),
#         'mfcc11_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[10]),
#         'mfcc12_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[11]),
#         'mfcc12_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[11]),
#         'mfcc13_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[12]),
#         'mfcc13_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[12]),
#         'mfcc14_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[13]),
#         'mfcc14_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[13]),
#         'mfcc15_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[14]),
#         'mfcc15_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[14]),
#         'mfcc16_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[15]),
#         'mfcc16_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[15]),
#         'mfcc17_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[16]),
#         'mfcc17_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[16]),
#         'mfcc18_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[17]),
#         'mfcc18_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[17]),
#         'mfcc19_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[18]),
#         'mfcc19_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[18]),
#         'mfcc20_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[19]),
#         'mfcc20_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[19]),
#     }

    
#     return pd.DataFrame([features])

# # Set up the Streamlit app
# st.title("Music Genre Classification")
# st.write("Upload a music file to predict its genre.")

# # Create a file uploader for audio files
# uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

# if uploaded_file is not None:
#     # Extract features from the uploaded audio file
#     with st.spinner('Extracting features...'):
#         # Save the uploaded file temporarily
#         temp_file_path = 'temp_audio_file'
#         with open(temp_file_path, 'wb') as f:
#             f.write(uploaded_file.read())
        
#         features_df = extract_features(temp_file_path)
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(features_df)
#         scaler_path = 'scaler.pkl'
#         joblib.dump(scaler, scaler_path)


#         if not features_df.empty:
#             st.write("Extracted Features:")
#             st.write(features_df)
#             st.write(features_df.shape)
#             # print(features_df.shape)  # Should match the training feature count (e.g., (n_samples, 52))

#         else:
#             st.error("No features extracted. Please check the audio file or extraction logic.")

    
#     # Predict the genre using the model
#     with st.spinner('Predicting genre...'):
#         prediction = model.predict(features_df)
    
#     # Display the prediction
#     st.write(f"The predicted genre is: {prediction[0]}")
#     predicted_genre = label_mapping.get(prediction[0], "Unknown Genre")

#     st.write(f"The predicted genre is: {predicted_genre}")


import joblib
import librosa
import numpy as np
import pandas as pd
import streamlit as st

# Load the pre-trained model
model_path = 'C:/Users/HP/OneDrive/Desktop/100DaysOfBytewise/project 2/rfc.pkl'
model = joblib.load(model_path)

# Load the scaler
scaler_path = 'C:/Users/HP/OneDrive/Desktop/100DaysOfBytewise/project 2/scaler.pkl'
scaler = joblib.load(scaler_path)

label_mapping = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
    4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop',
    8: 'reggae', 9: 'rock'
}

# Function to extract and scale features from the audio file
def extract_and_scale_features(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Extract features
    features = {
        'length': len(y),
        'chroma_stft_mean': np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
        'chroma_stft_var': np.var(librosa.feature.chroma_stft(y=y, sr=sr)),
        'rms_mean': np.mean(librosa.feature.rms(y=y)),
        'rms_var': np.var(librosa.feature.rms(y=y)),
        'spectral_centroid_mean': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_centroid_var': np.var(librosa.feature.spectral_centroid(y=y, sr=sr)),
        'spectral_bandwidth_var': np.var(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
        'zero_crossing_rate_var': np.var(librosa.feature.zero_crossing_rate(y=y)),
        'harmony_mean': np.mean(librosa.effects.harmonic(y)),
        'perceptr_mean': np.mean(librosa.feature.spectral_flatness(y=y)),
        'perceptr_var': np.var(librosa.feature.spectral_flatness(y=y)),
        'tempo': np.mean(librosa.beat.tempo(y=y, sr=sr)),
        'mfcc1_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[0]),
        'mfcc1_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[0]),
        'mfcc2_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[1]),
        'mfcc3_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[2]),
        'mfcc3_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[2]),
        'mfcc4_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[3]),
        'mfcc4_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[3]),
        'mfcc5_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[4]),
        'mfcc5_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[4]),
        'mfcc6_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[5]),
        'mfcc6_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[5]),
        'mfcc7_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[6]),
        'mfcc7_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[6]),
        'mfcc8_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[7]),
        'mfcc8_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[7]),
        'mfcc9_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[8]),
        'mfcc9_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[8]),
        'mfcc10_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[9]),
        'mfcc10_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[9]),
        'mfcc11_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[10]),
        'mfcc11_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[10]),
        'mfcc12_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[11]),
        'mfcc12_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[11]),
        'mfcc13_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[12]),
        'mfcc13_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[12]),
        'mfcc14_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[13]),
        'mfcc14_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[13]),
        'mfcc15_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[14]),
        'mfcc15_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[14]),
        'mfcc16_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[15]),
        'mfcc16_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[15]),
        'mfcc17_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[16]),
        'mfcc17_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[16]),
        'mfcc18_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[17]),
        'mfcc18_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[17]),
        'mfcc19_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[18]),
        'mfcc19_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[18]),
        'mfcc20_mean': np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[19]),
        'mfcc20_var': np.var(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)[19]),
    }

    features_df = pd.DataFrame([features])

    # Apply the loaded scaler to the extracted features
    features_scaled = scaler.transform(features_df)
    return features_scaled

# Set up the Streamlit app
st.title("Music Genre Classification")
st.write("Upload a music file to predict its genre.")

# Create a file uploader for audio files
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])

# Usage in your Streamlit app
if uploaded_file is not None:
    # Extract and scale features from the uploaded audio file
    with st.spinner('Extracting and scaling features...'):
        temp_file_path = 'temp_audio_file'
        with open(temp_file_path, 'wb') as f:
            f.write(uploaded_file.read())
        
        features_scaled = extract_and_scale_features(temp_file_path)
        
        if features_scaled.shape[1] == model.n_features_in_:
            st.write("Features have been extracted and scaled successfully.")
        else:
            st.error(f"Feature shape mismatch: Expected {model.n_features_in_}, got {features_scaled.shape[1]}")

    # Predict the genre using the scaled features
    with st.spinner('Predicting genre...'):
        prediction = model.predict(features_scaled)
    
    # Display the prediction
    predicted_genre = label_mapping.get(prediction[0], "Unknown Genre")
    st.write(f"The predicted genre is: {predicted_genre}")
