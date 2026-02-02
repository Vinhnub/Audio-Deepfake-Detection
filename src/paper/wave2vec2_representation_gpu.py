import soundfile as sf
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import soundfile as sf
import os
import torch
import pandas as pd
import numpy as np


class embedd_model:

  def __init__(self, processor, model, intermediate_embedding_layer = False, layer_index = False, pooling = False):
    self.processor = processor
    self.model = model
    self.pooling = pooling
    self.intermediate_embedding_layer = intermediate_embedding_layer
    self.layer_index = layer_index

    # ===== CUDA ADD =====
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.model = self.model.to(self.device)
    self.model.eval()
    # ===================


  def audio_to_representation(self, audio_file_path):
    data, sample_rate = sf.read(audio_file_path)

    if len(data.shape) > 1 and data.shape[1] > 1:
        print("Speech has multiple channel")
        data = data.mean(axis=1)

    float_array = data.astype(float)

    print("Audio file is converted to float type array of shape {}".format(float_array.shape))

    input_values = self.processor(
        float_array,
        return_tensors="pt"
    ).input_values.to(self.device)   # ===== CUDA ADD =====

    with torch.no_grad():  # ===== CUDA ADD =====
      if self.intermediate_embedding_layer:
        try:
          hidden_state = self.model(input_values).hidden_states[self.layer_index]
        except:
          print("################# Check the hidden layer index #################")  
          exit(1)
      else:
        hidden_state = self.model(input_values).last_hidden_state

    return hidden_state


  def extract_label(self, txt_file_path):
    with open(txt_file_path, 'r') as file:
      lines = file.readlines()
    data = [line.split() for line in lines]
    df = pd.DataFrame(data, columns=['id', 'file', 'dummy1', 'dummy2', 'label'])
    print(df)
    return df


  def complete_embedding(self, directory, label_file_path, get_label = True):
    files_in_directory = os.listdir(directory)
    recorded_audio_list = []

    file_names = [file for file in files_in_directory if os.path.isfile(os.path.join(directory, file))]

    count = 0
    df = pd.DataFrame(columns=[f"feature_{i}" for i in range(self.model.config.output_hidden_size)])

    if get_label:
      label_data = self.extract_label(label_file_path)
      self.y = pd.DataFrame(columns = ['label'])

    for file_name in file_names:
      count += 1
      print("id {}th file {} is processing ____________________________________".format(count, file_name))

      representation_layers = self.audio_to_representation(f"{directory}/{file_name}")

      if self.pooling:
        pass
      else:
        representation_layers = torch.mean(representation_layers[0], dim=0)

      # ===== CUDA ADD (.cpu()) =====
      row = pd.DataFrame(
          representation_layers.detach().cpu().numpy().reshape(1, -1),
          columns=df.columns
      )
      # ============================

      df = pd.concat([df, row], ignore_index=True)

      if get_label:
        try:
          label = label_data.loc[label_data['file']+'.flac' == file_name, 'label'].to_numpy().reshape(1)
          label = pd.DataFrame(label)
          label.columns = self.y.columns
          self.y = pd.concat([self.y, label])
          recorded_audio_list.append(file_name)
          print("################# got label #################")
        except:
          print("id {}th file {} is not processed ____________________________________".format(count, file_name))
          df.drop(df.index[-1], inplace=True)

    if get_label:
      self.y.reset_index(drop=True, inplace=True)
      recorded_audio_column = pd.DataFrame(recorded_audio_list, columns = ['file_name'])
      df = pd.concat([recorded_audio_column, df, self.y], axis = 1)

    return df



if __name__ == "__main__":

  processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
  model = Wav2Vec2Model.from_pretrained(
      "facebook/wav2vec2-base-960h",
      output_attentions=True,
      output_hidden_states=True
  )

  DATA_DIR = "D:/Pythonfile/Audio-Deepfake-Detection/data/raw/ASVspoof2019_LA_eval/flac"
  LABEL_PATH = r"D:/Pythonfile/Audio-Deepfake-Detection/data/raw/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"

  intermediate_embedding_layer = True
  layer_index = 3

  embedded_model = embedd_model(
      processor,
      model,
      intermediate_embedding_layer=intermediate_embedding_layer,
      layer_index=layer_index
  )

  df = embedded_model.complete_embedding(DATA_DIR, LABEL_PATH)

  df.to_csv(
      "D:/Pythonfile/Audio-Deepfake-Detection/data/feature/eval_intermediate_embedding_layer_"
      + str(intermediate_embedding_layer)
      + "_layer_index"
      + str(layer_index)
      + ".csv",
      index=False
  )
