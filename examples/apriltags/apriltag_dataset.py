import apriltag
import cv2
import os
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

_dir_path = os.path.dirname(os.path.realpath(__file__))
_data_path = os.path.join(_dir_path, '../../datasets/Apriltags-0')

class VideoDataset(Dataset):
    def __init__(self, video_file):
        self._video_file = video_file
        self._cap = cv2.VideoCapture(video_file)
        assert self._cap.isOpened(), video_file
        self._nframes = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def video_file():
        return self._video_file
    
    def __len__(self):
        return self._nframes
    
    def __getitem__(self, idx):
        assert self._cap.isOpened(), self._video_file
        assert idx < self._nframes, (self._video_file, idx, self._nframes)
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, image = self._cap.read()
        assert success, (self._video_file, idx)
        return image

class VideoFolderDataset(Dataset):
    def __init__(self, video_folder, pattern="*.mp4"):
        self.video_datasets = []
        self._total_frames = 0
        self._partition = []
        for video_file in glob.glob(os.path.join(video_folder,'*.mp4')):
            self.video_datasets.append(VideoDataset(video_file))
            self._partition.append(self._total_frames)
            self._total_frames += len(self.video_datasets[-1])

    def __len__(self):
        return self._total_frames
    
    def __getitem__(self, idx):
        for i, part in enumerate(self._partition):
            part_idx = idx - part
            assert part_idx >= 0
            if part_idx < len(self.video_datasets[i]):
                return self.video_datasets[i][part_idx]
        assert False, 'bad idx %d' % idx
        
def main():
    video_dataset = VideoFolderDataset(_data_path)
    print(len(video_dataset))
    detector = apriltag.Detector()    
    for image in video_dataset:
        cv2.imshow('image', image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        info = detector.detect(gray)
        print(info)
        #cv2.imshow('dimage', dimage)
        cv2.waitKey(10)

if __name__ == "__main__":
    main()
