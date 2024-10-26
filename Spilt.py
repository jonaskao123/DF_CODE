# @title Packages

import os
import random
import shutil
import mtcnn_utils
from tqdm import tqdm

# @title Split Real to Train and Test 

source_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real'
train_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-train'
test_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

video_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

random.shuffle(video_files)

train_videos = video_files[:472]
test_videos = video_files[472:]

for video in train_videos:
      shutil.copy(os.path.join(source_folder, video), os.path.join(train_folder, video))

for video in test_videos:
      shutil.copy(os.path.join(source_folder, video), os.path.join(test_folder, video))

# @title Split Synthesis to Train and Test

source_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis'
train_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-train'
test_folder = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-test'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

video_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

sampled_videos = random.sample(video_files, 590)
random.shuffle(sampled_videos)

train_videos = sampled_videos[:472]
test_videos = sampled_videos[472:]

for video in train_videos:
      shutil.copy(os.path.join(source_folder, video), os.path.join(train_folder, video))

for video in test_videos:
      shutil.copy(os.path.join(source_folder, video), os.path.join(test_folder, video))

# @title Face Detection for real

train_video_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-train'
train_output_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-train-frame'
train_video_files = os.listdir(train_video_dir)

for video_file in tqdm(train_video_files, desc='Processing videos'):
      video_path = os.path.join(train_video_dir, video_file)
      mtcnn_utils.process_and_save_faces(video_path, train_output_dir)

test_video_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-test'
test_output_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-real-test-frame'
test_video_files = os.listdir(test_video_dir)

for video_file in tqdm(test_video_files, desc='Processing videos'):
      video_path = os.path.join(test_video_dir, video_file)
      mtcnn_utils.process_and_save_faces(video_path, test_output_dir)

# @title Face Detection for synthesis

train_video_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-train'
train_output_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-train-frame'
train_video_files = os.listdir(train_video_dir)

for video_file in tqdm(train_video_files, desc='Processing videos'):
      video_path = os.path.join(train_video_dir, video_file)
      mtcnn_utils.process_and_save_faces(video_path, train_output_dir)

test_video_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-test'
test_output_dir = '/home/jonaskao/Data/Celeb-DF-v2/Celeb-synthesis-test-frame'
test_video_files = os.listdir(test_video_dir)

for video_file in tqdm(test_video_files, desc='Processing videos'):
      video_path = os.path.join(test_video_dir, video_file)
      mtcnn_utils.process_and_save_faces(video_path, test_output_dir)
