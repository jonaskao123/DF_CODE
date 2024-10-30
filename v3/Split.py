# @title Packages

import os
import utils
import random
import shutil
from tqdm import tqdm

def split_train_test_by_video(face_output_folder, train_folder, test_folder, train_ratio=0.8):
      if os.path.exists(train_folder):
            shutil.rmtree(train_folder)
      if os.path.exists(test_folder):
            shutil.rmtree(test_folder)

      os.makedirs(train_folder, exist_ok=True)
      os.makedirs(test_folder, exist_ok=True)

      video_prefixes = list(set([f.split('_', 2)[0] + '_' + f.split('_', 2)[1] for f in os.listdir(face_output_folder) if os.path.isfile(os.path.join(face_output_folder, f))]))
      # video_prefixes = list(set([f.split('_', 3)[0] + '_' + f.split('_', 3)[1] + '_' + f.split('_', 3)[2] for f in os.listdir(face_output_folder) if os.path.isfile(os.path.join(face_output_folder, f))]))

      random.shuffle(video_prefixes)
      train_count = int(len(video_prefixes) * train_ratio)
      train_prefixes = video_prefixes[:train_count]
      test_prefixes = video_prefixes[train_count:]

      for prefix in train_prefixes:
            for file in os.listdir(face_output_folder):
                  if file.startswith(prefix):
                        shutil.copy(os.path.join(face_output_folder, file), os.path.join(train_folder, file))

      for prefix in test_prefixes:
            for file in os.listdir(face_output_folder):
                  if file.startswith(prefix):
                        shutil.copy(os.path.join(face_output_folder, file), os.path.join(test_folder, file))

face_output_folder = '/home/jonaskao/Data/v3/Celeb-real-faces'
train_folder = '/home/jonaskao/Data/v3/Celeb-real-train'
test_folder = '/home/jonaskao/Data/v3/Celeb-real-test'

split_train_test_by_video(face_output_folder, train_folder, test_folder)
