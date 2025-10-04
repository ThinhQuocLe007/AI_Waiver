import pyaudio
import soundfile as sf
import time
import wave
import speech_recognition as sr 
from typing import Optional


# NOTE: speech_recognition is no longer needed for the time-based recording method.
# It would only be needed if you wanted to add back a VAD-based recording method.

class Microphone:
    def __init__(self, mic_index: int = 4, sample_rate: int = 44100):
        """
        Initializes the Microphone utility class.
        
        Args:
            mic_index (int): The index of the microphone to use. Run list_all_devices() to see options.
            sample_rate (int): The sample rate to capture audio at.
        """
        self.format = pyaudio.paInt16      # 16-bit resolution
        self.channels = 1                  # Mono
        self.sample_rate = sample_rate            # Samples per second
        self.chunk = 1024                  # Samples per frame
        self.mic_index = mic_index
        self.audio_interface = pyaudio.PyAudio()

    def list_all_devices(self):
        """
        Lists all available audio input devices found by PyAudio.
        """
        print("\n--- Available Audio Input Devices ---")
        device_count = self.audio_interface.get_device_count()
        if device_count == 0:
            print("No audio devices found.")
            return

        for i in range(device_count):
            try:
                device_info = self.audio_interface.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:
                    print(f"Device Index: {i}")
                    print(f"  Name: {device_info.get('name')}")
                    print(f"  Max Input Channels: {device_info.get('maxInputChannels')}")
                    print(f"  Default Sample Rate: {int(device_info.get('defaultSampleRate'))} Hz\n")
            except Exception as e:
                print(f"Could not get info for device index {i}: {e}")
        print("-----------------------------------")

    
    def inspect_audio(self, file_path: str):
        """
        Inspects an audio file and prints its key properties.

        Args:
            file_path (str): The path to the audio file.
        """
        try:
            info = sf.info(file_path)
            print("\n--- Audio File Information ---")
            print(f"File Path:    {file_path}")
            print(f"Sample Rate:  {info.samplerate} Hz")
            print(f"Channels:     {info.channels}")
            print(f"Duration:     {info.duration:.2f} seconds")
            print(f"Format:       {info.format_info}")
            print("----------------------------")
        except Exception as e:
            print(f"❌ Error inspecting file: {e}")
            print("Please ensure the file path is correct and it's a valid audio file.")

    def record(self, duration: int = 5) -> str:
        """
        Records audio from the microphone for a fixed duration.
        The filename is automatically generated based on the current timestamp.

        Args:
            duration (int): The number of seconds to record for.
        
        Returns:
            str: The filename of the saved audio.
        """
        print(f"\nPreparing to record for {duration} seconds...")
        
        stream = self.audio_interface.open(
            format=self.format,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.mic_index
        )
        
        print("🔴 Recording started...")
        
        frames = []
        # Loop to record audio chunk by chunk for the specified duration
        for _ in range(0, int(self.sample_rate / self.chunk * duration)):
            # --- THE FIX: Add exception_on_overflow=False to stream.read() ---
            # This tells the stream to not crash if it overflows.
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            
        print("✅ Recording finished.")
        
        # Stop and close the audio stream
        stream.stop_stream()
        stream.close()
        
        # Generate filename with timestamp
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"recording_at_{timestamp}.wav"

        # Save the recorded data as a WAV file
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self.audio_interface.get_sample_size(self.format))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(frames))
            
        print(f"Audio successfully saved to {file_path}")
        return file_path


    # Record with VAD by recognition lib 
    def record_with_vad(self, audio_path: str = 'output.wav', timeout: int = 10): 
        """
        Consider build record with VAD using speech recognition lib 
        Args: 
            - audio_path (str) where save the audio 
            - timeout (str) how much time to waiting for 1 speech 
        
        """
        recognizer = sr.Recognizer() 

        source = sr.Microphone(device_index= self.mic_index,sample_rate= self.sample_rate )

        # Calibrate for ambient noise         
        print('Calibrating gfor ambient noise, please wait...')
        with source as mic: # Wait, whaht ?? 
            recognizer.adjust_for_ambient_noise(mic)
            print('Calibration complete. Listening for speech....')

         
            try: 
                audio_data = recognizer.listen(mic, timeout= timeout)
                print('Speech detected! Saving the recording...')

                with open(audio_path, 'wb') as file: 
                    file.write(audio_data.get_wav_data())
                
                print(f'Audio sucessfully save to: {audio_path}')
                return audio_path

            except sr.WaitTimeoutError as e: 
                print(f'No speech detected within the timeout period')
                return None 
            except Exception as e: 
                print(f'An error occurred: {e}')
                return None 