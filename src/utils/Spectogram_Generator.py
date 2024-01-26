import pickle
import numpy as np
import matplotlib.pyplot as plt
import os

class GenerateImages():
  
  def __init__(self,data_path): 
       
    with open(data_path, 'rb') as file:
      self.loaded_data = pickle.load(file, encoding='latin1')
      
    self.sampling_rate = 100.0
      
  @staticmethod    
  def plot_spectral_analysis_spectrogram(signal,sampling_rate,file_name):
    # Show Spectogram
    plt.figure(figsize=(10, 6))
    plt.specgram(signal, Fs=sampling_rate, cmap='viridis', aspect='auto')
    plt.savefig(file_name)
    plt.close("all")
    
  def generate_image(self,signal_tuple):
    for i in range(len(self.loaded_data[signal_tuple])):
      componente_Q = self.loaded_data[signal_tuple][i][0]
      componente_I = self.loaded_data[signal_tuple][i][1]

      Q = componente_Q
      I = componente_I

      # Complex Signal
      segnale_complesso = I + 1j * Q

      # Normalization
      segnale_complesso_normalized = segnale_complesso / np.max(np.abs(segnale_complesso))
      dir = f"{signal_tuple[0]}_{signal_tuple[1]}"
      try:
        os.mkdir(dir)
      except:
        ...
        
      GenerateImages.plot_spectral_analysis_spectrogram(segnale_complesso, self.sampling_rate, f"{dir}/{i+1}.png")

      print(f"Saved {i+1}/{len(self.loaded_data[signal_tuple])}")

    os.system(f"zip {dir}.zip {dir}/*")
    os.system(f"rm -rf {dir}")
  
  def generate_images(self):
    for signal_tuple in self.loaded_data:
      self.generate_image(signal_tuple=signal_tuple)    
    
      
  
generator = GenerateImages(data_path="../../../TEST/RML2016.10a_dict.pkl")
generator.generate_images()
