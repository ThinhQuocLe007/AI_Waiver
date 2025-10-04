import numpy as np 
import torch 
import pyaudio
import time 
import wave 

class Silero_VAD: 
    def __init__(self, sample_rate = 16000, chunk_size=512): 
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        print("Loading Silero VAD model...")
        self.model, _  = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                        model='silero_vad', 
                                        force_reload=False) # <-- FIX #2: Prevents re-downloading

        self.audio_interface = pyaudio.PyAudio() 
        print("VAD initialized successfully.")

    def listen(self, silence_chunks_needed=8): 
        # create a stream 
        stream = self.audio_interface.open(
            format=pyaudio.paInt16, 
            channels=1, 
            rate=self.sample_rate, 
            input=True, 
            frames_per_buffer=self.chunk_size
        )

        # listen loop 
        print('\n🎤 Listening for speech...')
        recorded_frames = [] 
        is_speaking = False 
        silence_counter = 0 
        
        while True: 
            audio_chunk = stream.read(self.chunk_size)
            audio_int16 = torch.from_numpy(np.frombuffer(audio_chunk, dtype=np.int16))
            audio_float32 = audio_int16.to(torch.float32) / 32768.0 

            speech_confidence = self.model(audio_float32, self.sample_rate).item() 
            
            if speech_confidence > 0.5: 
                if not is_speaking: 
                    print("   (Speech started...)")
                    is_speaking = True 
                silence_counter = 0 
                recorded_frames.append(audio_chunk)
            
            elif is_speaking:
                silence_counter += 1 
                recorded_frames.append(audio_chunk)
                if silence_counter > silence_chunks_needed: 
                    print("   (Speech ended due to pause.)")
                    break 
            
        stream.stop_stream() 
        stream.close() 

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"silero_vad_{timestamp}.wav"
        
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.audio_interface.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.sample_rate)
            wf.writeframes(b''.join(recorded_frames))
            
        print(f"✅ Recording saved to: {file_path}")
        return file_path