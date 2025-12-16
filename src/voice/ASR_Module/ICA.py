import numpy as np 
import soundfile as sf
from sklearn.decomposition import FastICA
import matplotlib.pyplot as plt

class ICASeparator: 
    def __init__(self, n_components: int = 2):
        """
        Initializes the ICASeparator with the specified number of components.

        Args:
            n_components (int): Number of components to extract. If None, all components are used.
        """
        self.n_components = n_components
    
    def load_audio(self, file_path: str):
        """
        Loads an audio file and returns the audio data and sample rate.

        Args:
            file_path (str): The path to the audio file.

        Returns:
            tuple: A tuple containing the audio data (numpy array) and the sample rate (int).
        """
        try: 
            audio_data, sample_rate = sf.read(file_path)
            return audio_data, sample_rate
        except Exception as e: 
            print(f"❌ Error loading audio file: {e}")
            raise 

    
    def perform_ica(self, mixed_audio): 
        """
        Perform ICA on the mixed audio signals.
        """
        try: 
            # Ensure mixed_audio is 2D
            if len(mixed_audio.shape) == 1:
                mixed_audio = mixed_audio.reshape(-1, 1)
            
            # Apply ICA
            ica = FastICA(n_components=self.n_components)
            separated_signal = ica.fit_transform(mixed_audio)
            return separated_signal
        except Exception as e:
            print(f"❌ Error performing ICA: {e}")
            raise 

    def save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str):
        """
        Saves the audio data to a file.

        Args:
            audio_data (np.ndarray): The audio data to save.
            sample_rate (int): The sample rate of the audio data.
            output_path (str): The path to save the audio file.
        """
        try: 
            for i in range(self.n_components):
                output_path = f'{output_path}_component_{i+1}.wav'

                # Normalize the separated signal
                separated_signal = audio_data[:, i]
                separated_signal = separated_signal / np.max(np.abs(separated_signal))
                sf.write(output_path, separated_signal, sample_rate)
                print(f'Separated audio component {i+1} saved to {output_path}')

        except Exception as e:
            print(f"❌ Error saving audio file: {e}")
            raise
        


    def plot_signals(self, mixed_audio, separated_signals):
        """
        Plot the mixed and separated signals.
        
        Args:
            mixed_audio (np.ndarray): Mixed audio signals.
            separated_signals (np.ndarray): Separated audio signals.
        """
        plt.figure(figsize=(12, 8))
        
        # Plot mixed signals
        plt.subplot(3, 1, 1)
        plt.plot(mixed_audio)
        plt.title("Mixed Audio Signal")
        
        # Plot separated signals
        for i, signal in enumerate(separated_signals.T):
            plt.subplot(3, 1, i + 2)
            plt.plot(signal)
            plt.title(f"Separated Component {i+1}")
        
        plt.tight_layout()
        plt.show()

    def process(self, input_file: str, output_prefix = 'separated_'):
        """
        Processes the input audio file to separate sources using ICA and saves the results.

        Args:
            input_file (str): The path to the input audio file.
            output_prefix (str): The prefix for the output separated audio files.
        """
        try: 
            # Load audio
            audio_data, sample_rate = self.load_audio(input_file)

            # Perform ICA
            separated_signals = self.perform_ica(audio_data)
            # Check the shape of the separated signals
            print(f"Separated signals shape: {separated_signals.shape}")
            print(f"Number of samples: {separated_signals.shape[0]}")
            print(f"Number of components: {separated_signals.shape[1]}")

            # Save separated audio files
            self.save_audio(separated_signals, sample_rate, output_prefix)

            # Plot signals
            self.plot_signals(audio_data, separated_signals)

        except Exception as e: 
            print(f"❌ Error in processing: {e}")
            raise