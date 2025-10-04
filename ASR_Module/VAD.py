import speech_recognition as sr
import time
import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def main():
    # --- 1. Initialize PhoASR Model ---
    print("Loading PhoASR model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "vinai/PhoWhisper-base"
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        print(f"✅ PhoASR model loaded on {device}.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- 2. Initialize Recognizer and Microphone ---
    r = sr.Recognizer()
    mic = sr.Microphone(device_index=4) # Use 16kHz as it's Whisper's native rate

    # --- VAD PARAMETER TUNING ---
    r.pause_threshold = 0.8  # seconds of non-speaking audio before a phrase is considered complete
    r.phrase_time_limit = 15 # max seconds a phrase can be
    
    # --- 3. Calibrate for ambient noise ---
    with mic as source:
        print("\nCalibrating for ambient noise... Please be quiet.")
        r.adjust_for_ambient_noise(source, duration=1.5)
        print(f"Calibration complete. Energy threshold: {r.energy_threshold:.2f}")

    # --- 4. Main Loop: Listen -> Save -> Transcribe ---
    while True:
        try:
            with mic as source:
                print("\nListening for your next command...")
                
                # The listen() method is the VAD. It blocks until speech is detected.
                audio = r.listen(source, timeout=10)

                print("Speech detected! Processing...")
                
                # --- a) Save the detected audio to a WAV file ---
                timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                filename = f"audio_at_{timestamp}.wav"
                with open(filename, "wb") as f:
                    f.write(audio.get_wav_data())
                print(f"Audio saved as {filename}")

                # --- b) Transcribe the audio using PhoASR ---
                # Get raw audio data from the AudioData object
                raw_data = audio.get_raw_data()
                
                # Convert raw data to a NumPy array
                audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Process and transcribe
                input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features.to(device)
                predicted_ids = model.generate(input_features)
                transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                
                print(f"--> [Transcription]: {transcription}")
                print("Processing complete. Ready for next command.")
                
        except sr.WaitTimeoutError:
            print("Timeout: No speech detected. Listening again...")
        except KeyboardInterrupt:
            print("\nExiting program.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

if __name__ == "__main__":
    main()