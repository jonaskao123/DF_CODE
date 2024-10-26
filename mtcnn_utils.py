import os
import cv2
import sys
from mtcnn import MTCNN
from contextlib import contextmanager

@contextmanager
def suppress_output():
      with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr

            try:
                  sys.stdout = fnull
                  sys.stderr = fnull
                  yield
            finally:
                  sys.stdout = old_stdout
                  sys.stderr = old_stderr

detector = MTCNN()

def save_faces_from_frame(video_name, frame, frame_number, output_dir):
      with suppress_output():
            faces = detector.detect_faces(frame)

      if not os.path.exists(output_dir):
            os.makedirs(output_dir)

      face_files = []

      for i, face in enumerate(faces):
            x, y, w, h = face['box']
            face_region = frame[y:y+h, x:x+w]

            face_filename = f"{video_name}_frame{frame_number}_face{i}.jpg"
            face_path = os.path.join(output_dir, face_filename)
            cv2.imwrite(face_path, face_region)

            face_files.append(face_path)

      return face_files

def process_and_save_faces(video_path, output_dir, start_frame=241, end_frame=320, frame_interval=8, max_images=10):
      cap = cv2.VideoCapture(video_path)
      fps = int(cap.get(cv2.CAP_PROP_FPS))
      video_name = os.path.splitext(os.path.basename(video_path))[0]

      face_file_paths = []

      if fps == 0:
            cap.release()
            return None
      
      frame_count = 0
      
      for frame_number in range(start_frame, end_frame+1, frame_interval):
            if frame_count >= max_images:
                  break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()

            if ret:
                  face_files = save_faces_from_frame(video_name, frame, frame_number, output_dir)
                  face_file_paths.extend(face_files)
                  frame_count += 1
            else:
                  break
            
      cap.release()
      return face_file_paths
