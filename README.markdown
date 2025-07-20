# English Accent Detection AI Model

## Overview
This project features an AI model designed to detect the regional or cultural accent of a person speaking English. The model analyzes audio samples to classify the speaker's accent based on linguistic patterns, phonetic characteristics, and speech nuances.

## Dataset
The model is trained on the **Speech Accent Archive** dataset, available on Kaggle: [Speech Accent Archive](https://www.kaggle.com/datasets/rtatman/speech-accent-archive). This dataset contains audio recordings of English speakers from diverse linguistic and cultural backgrounds, each reading a standardized passage. It includes metadata such as the speaker's native language, region, and other demographic details.

## Features
- **Accent Classification**: Identifies the speaker's accent (e.g., British, American, Indian, Australian, etc.) based on audio input.
- **High Accuracy**: Utilizes a Random Forest Classifier for robust performance across various accents.
- **API Integration**: Includes a FastAPI-based interface for real-time accent detection via audio uploads.
- **Scalable**: Can be extended to support additional accents with further training.

## Model Architecture
The model uses a machine learning approach with the following components:
- **Preprocessing**: Audio files are processed using Librosa to extract MFCC (Mel-frequency cepstral coefficients) features.
- **Machine Learning Model**: A Random Forest Classifier trained on preprocessed audio features.
- **Output**: Predicts the accent category for a given audio input.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/guigcr/accent-detection-model.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/rtatman/speech-accent-archive) and place it in the `data/` directory.

## Usage
1. Preprocess the dataset:
   ```bash
   python preprocess.py --data_path data/speech-accent-archive
   ```
2. Train the model:
   ```bash
   python train.py --model random_forest --n_estimators 100
   ```
3. Run the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
4. Test the model via API:
   - Send a POST request to `http://localhost:8000/predict` with an audio file (e.g., WAV format).

## Requirements
- Python 3.8+
- Libraries: 
  - `numpy`
  - `librosa`
  - `pandas`
  - `scikit-learn`
  - `joblib`
  - `fastapi`
  - `uvicorn`
  - `tempfile`
  - See `requirements.txt` for a complete list.

## API Endpoints
- **POST /predict**: Upload an audio file to receive the predicted accent.
  - Request: `multipart/form-data` with a `.wav` file.
  - Response: JSON containing the predicted accent and confidence score.

## Future Work
- Enhance the model with deep learning techniques (e.g., CNN or RNN) for improved accuracy.
- Expand the API to support real-time streaming audio.
- Incorporate additional datasets for broader accent coverage.
- Optimize for low-resource devices.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss improvements or bugs.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.