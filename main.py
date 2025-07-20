import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import uvicorn
import warnings
warnings.filterwarnings('ignore')

class AccentDetectorCustom:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.accent_mapping = {}
        
    def extract_features(self, audio_file_path, max_duration=10):
        """
        Extract comprehensive audio features from an audio file
        """
        try:
            # Load audio file with limited duration for consistency
            y, sr = librosa.load(audio_file_path, sr=22050, duration=max_duration)
            
            if len(y) == 0:
                return None
                
            features = []
            
            # MFCC features (13 coefficients)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend([
                np.mean(mfccs, axis=1).flatten(),
                np.std(mfccs, axis=1).flatten(),
                np.max(mfccs, axis=1).flatten(),
                np.min(mfccs, axis=1).flatten()
            ])
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features.extend([
                np.mean(chroma, axis=1),
                np.std(chroma, axis=1),
                np.max(chroma, axis=1),
                np.min(chroma, axis=1)
            ])
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            
            features.extend([
                [np.mean(spectral_centroids), np.std(spectral_centroids), np.max(spectral_centroids), np.min(spectral_centroids)],
                [np.mean(spectral_rolloff), np.std(spectral_rolloff), np.max(spectral_rolloff), np.min(spectral_rolloff)],
                [np.mean(spectral_bandwidth), np.std(spectral_bandwidth), np.max(spectral_bandwidth), np.min(spectral_bandwidth)]
            ])
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            features.append([np.mean(zcr), np.std(zcr), np.max(zcr), np.min(zcr)])
            
            # Tempo and rhythm
            try:
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
                features.append([tempo, len(beats)])
            except:
                features.append([120.0, 0])  # Default values
            
            # Pitch features
            try:
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                pitches = pitches[pitches > 0]
                if len(pitches) > 0:
                    features.append([np.mean(pitches), np.std(pitches), np.max(pitches), np.min(pitches)])
                else:
                    features.append([0, 0, 0, 0])
            except:
                features.append([0, 0, 0, 0])
            
            # Mel-scale spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            features.extend([
                [np.mean(mel_spec), np.std(mel_spec), np.max(mel_spec), np.min(mel_spec)]
            ])
            
            # Flatten all features
            feature_vector = []
            for feature_group in features:
                if isinstance(feature_group, list):
                    for item in feature_group:
                        if isinstance(item, np.ndarray):
                            feature_vector.extend(item.flatten())
                        else:
                            feature_vector.append(item)
                elif isinstance(feature_group, np.ndarray):
                    feature_vector.extend(feature_group.flatten())
                else:
                    feature_vector.append(feature_group)
            
            return np.array(feature_vector, dtype=np.float32)
            
        except Exception as e:
            print(f"Error processing {audio_file_path}: {e}")
            return None
    
    def extract_accent_from_filename(self, filename):
        """
        Extract accent/language from filename (e.g., 'afrikaans5.mp3' -> 'afrikaans')
        """
        # Remove file extension
        name = os.path.splitext(filename)[0]
        
        # Extract language/accent part (remove numbers at the end)
        accent = re.sub(r'\d+$', '', name).lower()
        
        return accent
    
    def load_and_prepare_dataset(self, recordings_dir, speakers_csv_path=None):
        """
        Load dataset from your specific structure
        """
        features = []
        accents = []
        
        print(f"Loading audio files from: {recordings_dir}")
        
        # Get all audio files from recordings directory
        audio_files = [f for f in os.listdir(recordings_dir) 
                      if f.lower().endswith(('.mp3', '.wav', '.flac', '.m4a'))]
        
        print(f"Found {len(audio_files)} audio files")
        
        # Load speakers data if available
        speaker_info = {}
        if speakers_csv_path and os.path.exists(speakers_csv_path):
            try:
                df = pd.read_csv(speakers_csv_path)
                print(f"Loaded speaker information: {df.columns.tolist()}")
                # Create mapping if there are useful columns
                for idx, row in df.iterrows():
                    speaker_info[idx] = row.to_dict()
            except Exception as e:
                print(f"Could not load speakers CSV: {e}")
        
        successful_extractions = 0
        
        for audio_file in audio_files:
            print(f"Processing: {audio_file}")
            
            file_path = os.path.join(recordings_dir, audio_file)
            
            # Extract features
            feature_vector = self.extract_features(file_path)
            
            if feature_vector is not None:
                # Extract accent from filename
                accent = self.extract_accent_from_filename(audio_file)
                
                features.append(feature_vector)
                accents.append(accent)
                successful_extractions += 1
                
                if successful_extractions % 10 == 0:
                    print(f"Processed {successful_extractions} files successfully")
        
        print(f"\nDataset preparation complete!")
        print(f"Successfully processed: {successful_extractions} files")
        
        if len(features) == 0:
            print("No features extracted! Check your audio files.")
            return None, None
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(accents)
        
        # Show accent distribution
        unique_accents, counts = np.unique(y, return_counts=True)
        print(f"\nAccent distribution:")
        for accent, count in zip(unique_accents, counts):
            print(f"  {accent}: {count} samples")
        
        return X, y
    
    def train_model(self, recordings_dir, speakers_csv_path=None):
        """
        Train the accent detection model
        """
        print(" Starting model training...")
        
        # Load and prepare dataset
        X, y = self.load_and_prepare_dataset(recordings_dir, speakers_csv_path)
        
        if X is None or len(X) == 0:
            print(" No data to train on!")
            return False
        
        print(f" Dataset shape: {X.shape}")
        print(f" Number of features per sample: {X.shape[1]}")
        
        # Filter out classes with fewer than 2 samples
        unique_accents, counts = np.unique(y, return_counts=True)
        valid_classes = unique_accents[counts >= 2]
        valid_mask = np.isin(y, valid_classes)
        
        # Apply filter to X and y
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            print(" No valid data after filtering classes with fewer than 2 samples!")
            return False
        
        print(f" Filtered dataset shape: {X.shape}")
        print(f" Filtered accent distribution:")
        unique_accents, counts = np.unique(y, return_counts=True)
        for accent, count in zip(unique_accents, counts):
            print(f"  {accent}: {count} samples")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Create accent mapping for easy lookup
        self.accent_mapping = {
            i: accent for i, accent in enumerate(self.label_encoder.classes_)
        }
        
        print(f" Accent classes: {list(self.accent_mapping.values())}")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        print("Taining Random Forest model...")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n Training Complete!")
        print(f" Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        
        # Detailed report - CORREÇÃO AQUI:
        print(f"\n Detailed Classification Report:")
        
        # Método mais seguro - usa apenas classes presentes no teste
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Save model
        self.save_model()
        
        return True
    
    def predict_accent(self, audio_file_path):
        """
        Predict accent from audio file
        """
        if self.model is None:
            return None
        
        # Extract features
        features = self.extract_features(audio_file_path)
        if features is None:
            return None
        
        # Scale features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Create results
        results = {}
        for i, prob in enumerate(probabilities):
            accent = self.accent_mapping[i]
            results[accent] = float(prob * 100)  # Convert to percentage
        
        # Sort by probability
        results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
        
        return results
    
    def save_model(self):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'accent_mapping': self.accent_mapping
        }
        
        joblib.dump(model_data, 'accent_detection_model.pkl')
        print("💾 Model saved as 'accent_detection_model.pkl'")
    
    def load_model(self):
        """Load pre-trained model"""
        try:
            model_data = joblib.load('accent_detection_model.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.accent_mapping = model_data['accent_mapping']
            print("Model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Initialize FastAPI app
app = FastAPI(title="Accent Detection API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance
detector = AccentDetectorCustom()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    if os.path.exists('accent_detection_model.pkl'):
        detector.load_model()
        print("I started with pre-trained model")
    else:
        print("I started - no model found, train first!")

@app.get("/")
async def root():
    return {"message": "Accent Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": detector.model is not None}

@app.post("/train")
async def train_model(recordings_dir: str, speakers_csv: str = None):
    """Train the model with your dataset"""
    try:
        if not os.path.exists(recordings_dir):
            raise HTTPException(status_code=400, detail=f"Recordings directory not found: {recordings_dir}")
        
        success = detector.train_model(recordings_dir, speakers_csv)
        
        if success:
            return {"success": True, "message": "Model trained successfully!"}
        else:
            return {"success": False, "message": "Training failed"}
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict_accent(file: UploadFile = File(...)):
    """Predict accent from uploaded audio file"""
    try:
        if detector.model is None:
            raise HTTPException(status_code=400, detail="No model loaded. Train the model first.")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Predict accent
            results = detector.predict_accent(temp_file_path)
            
            if results:
                return {
                    "success": True,
                    "filename": file.filename,
                    "predictions": results,
                    "top_prediction": max(results, key=results.get)
                }
            else:
                return {"success": False, "error": "Could not analyze audio file"}
                
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/accents")
async def get_available_accents():
    """Get list of accents the model can detect"""
    if detector.model is not None and detector.accent_mapping:
        return {
            "accents": list(detector.accent_mapping.values()),
            "count": len(detector.accent_mapping)
        }
    else:
        return {"accents": [], "count": 0}

# Training script function
def train_from_command_line():
    """Function to train model from command line with hardcoded paths"""
    print("ACCENT DETECTION MODEL TRAINER")
    print("=" * 50)
    
    recordings_dir = r"C:\Users\Administrator\Desktop\1m\s\data\recordings"
    speakers_csv = r"C:\Users\Administrator\Desktop\1m\s\data\speakers_all.csv"
    
    if not os.path.exists(recordings_dir):
        print(f"Recordings directory not found: {recordings_dir}")
        return
    
    if not os.path.exists(speakers_csv):
        print(f" Speakers CSV not found: {speakers_csv}, continuing without it...")
        speakers_csv = None
    
    detector = AccentDetectorCustom()
    success = detector.train_model(recordings_dir, speakers_csv)
    
    if success:
        print("\n Training completed successfully!")
        print("You can now run the API with: uvicorn main:app --reload")
    else:
        print("\n Training failed!")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_from_command_line()
    else:
        print("Sarting FastAPI server...")
        print("Options:")
        print("  python main.py train  - Train the model")
        print("  uvicorn main:app --reload  - Start API server")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
