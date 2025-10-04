import torch
import librosa
import argparse
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from typing import Optional

class PhoASR:
    """
    A class to handle Vietnamese speech-to-text transcription using PhoWhisper models.
    """
    def __init__(self, model_name: str = "vinai/PhoWhisper-base"):
        """
        Initializes the PhoASR transcriber.

        This method loads the specified PhoWhisper model and processor from Hugging Face
        and prepares them for transcription, automatically selecting the best available device (GPU or CPU).

        Args:
            model_name (str): The name of the PhoWhisper model to use from Hugging Face.
                              Examples: "vinai/PhoWhisper-small", "vinai/PhoWhisper-base".
        """
        print(f"--- Initializing PhoASR with model: '{model_name}' ---")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.processor = None
        self.model = None

        try:
            self.processor = WhisperProcessor.from_pretrained(self.model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            print(f"✅ Model loaded successfully on device: '{self.device}'.")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Please check the model name and your internet connection.")
            # Set model to None to prevent usage if initialization fails
            self.model = None

    def transcribe(self, audio_path: str) -> Optional[str]:
        """
        Transcribes an audio file into Vietnamese text.

        Args:
            audio_path (str): The path to the audio file (e.g., 'recording.wav', 'song.mp3').

        Returns:
            Optional[str]: The transcribed text as a string, or None if an error occurs.
        """
        if not self.model or not self.processor:
            print("❌ Model not initialized. Cannot transcribe.")
            return None

        print(f"\n--- Processing Audio File: {audio_path} ---")
        try:
            # Load the audio file. librosa automatically resamples it to 16,000 Hz,
            # which is required by the Whisper model.
            speech_array, sampling_rate = librosa.load(audio_path, sr=16000)
            print(f"Audio loaded and resampled to {sampling_rate} Hz.")
        except Exception as e:
            print(f"❌ Error loading audio file: {e}")
            print("Please check the file path and ensure it is a valid audio format.")
            return None
            
        print("Transcribing... (This may take a moment)")
        
        # Preprocess the audio waveform to create input features for the model
        input_features = self.processor(
            speech_array, 
            sampling_rate=16000, 
            return_tensors="pt"
        ).input_features.to(self.device)
        
        # Generate the sequence of token IDs from the input features
        # We explicitly set the task and language for better performance and to avoid warnings.
        predicted_ids = self.model.generate(
            input_features, 
            task="transcribe", 
            language="vi"
        )
        
        # Decode the token IDs back into a human-readable text string
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        print("✅ Transcription complete.")
        return transcription
