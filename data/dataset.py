import os
import json
import argparse
import moviepy.editor as mp
import numpy as np
import itertools

import torch

from PIL import Image
from skimage import io, img_as_ubyte
from tqdm import tqdm


class VoxCelebDataset(object):
    """
    Class for managing and processing VoxCeleb2 video dataset.
    """
    def __init__(self, root):
        """
        :param root: dataset folder. The structure should be:
            root -> speaker directories -> video directories -> sliced videos
        """
        self.root = root
        self.metadata_ = self._construct_metadata(self.root)
    
    @property
    def metadata(self):
        return self.metadata_
    
    def _construct_metadata(self, root):
        metadata = dict()
        metadata['abspath'] = os.path.abspath(root)
        metadata['content'] = {}
        for speaker_folder in sorted(os.listdir(root)):
            metadata['content'][speaker_folder] = dict()
            video_folders = sorted(os.listdir(f'{root}/{speaker_folder}'))
            for video_folder in video_folders:
                metadata['content'][speaker_folder][video_folder] = []
                videos = sorted(os.listdir(f'{root}/{speaker_folder}/{video_folder}'))
                for filename in videos:
                    metadata['content'][speaker_folder][video_folder].append(filename)
        return metadata
    
    def _extract_frames_from_video(self, filename, to_dir, resize_shape, step=2):
        hierarchy = filename.split('/')
        path = f"{to_dir}/{'/'.join(hierarchy[-3:-1])}/{hierarchy[-1].split('.')[0]}"
        if not os.path.exists(path):
            os.makedirs(path)
        clip = mp.VideoFileClip(filename)
        clip_resized = clip.resize(height=resize_shape[0], width=resize_shape[1])
        for frame_idx, frame in enumerate(clip_resized.iter_frames()):
            if frame_idx % step == 0:
                frame_rgb = Image.fromarray(frame)
                frame_rgb.save(f'{path}/{frame_idx}.PNG')
        
    def preprocess_videos(self, to_dir='./img_dataset', resize_shape=(128, 128), frame_step=2):
        if os.path.exists(to_dir):
            raise RuntimeError(f'Directory {os.path.abspath(to_dir)} already exists. Remove it before preprocessing video data.')
        os.makedirs(name=to_dir)
        with open(f'{to_dir}/metadata.json', 'w') as f:
            json.dump(self.metadata, f)
        
        metadata = self.metadata['content']
        for speaker_folder in tqdm(metadata.keys()):
            for video_folder in metadata[speaker_folder].keys():
                for filename in metadata[speaker_folder][video_folder]:
                    filename = f'{self.root}/{speaker_folder}/{video_folder}/{filename}'
                    self._extract_frames_from_video(filename, to_dir, resize_shape, step=frame_step)


class SingleFaceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        center_identity_size=8,
        center_identity_step=32,
        transform=None,
        transform_ci=None,
        size=1600
    ):
        """
        :param root: str, speaker path, where his videos are laying. Folder hierarchy should be:
            video folders -> slice folders -> frames and keypoints
        :param center_identity_size: number of consequent frames to use to calculate center identity
        :param center_identity_step: number of iterations until CIs regeneration
        :param transform: transformations to apply to target frame and its keypoints
        :param transform_ci: transformations to apply to CI frames
        :param size: preferable number of iterations per epoch
        """
        super(SingleFaceDataset, self).__init__()
        self.root = root
        self.center_identity_size = center_identity_size
        self.center_identity_step = center_identity_step
        self.transform = transform
        self.transform_ci = transform_ci
        self.size = size
        self.metadata_, self.filelist_ = self._construct_metadata(self.root)
        self._videos = list(self.metadata_.keys())
        self._cis = self._generate_center_identities()
        self._cis_counter = 1

    @property
    def metadata(self):
        """
        Contains dict of total images hierarchy with the structure:
            video folders -> slice folders -> frames and keypoints
        """
        return self.metadata_
    
    @property
    def filelist(self):
        return self.filelist_

    def _construct_metadata(self, root):
        metadata = dict()
        filelist = list()
        video_folders = sorted(os.listdir(f'{root}'))
        for video_folder in video_folders:
            metadata[video_folder] = dict()
            slice_folders = sorted(os.listdir(f'{root}/{video_folder}'))
            for slice_folder in slice_folders:
                metadata[video_folder][slice_folder] = list()
                filenames = sorted(
                    os.listdir(f'{root}/{video_folder}/{slice_folder}'),
                    key=lambda filename: int(filename.split('.')[0])
                )
                for filename in filenames:
                    if '.PNG' in filename:
                        metadata[video_folder][slice_folder].append(filename)
                        img_path = f'{root}/{video_folder}/{slice_folder}/{filename}'
                        filelist.append(img_path)
                        assert os.path.exists(img_path.replace('.PNG', '.pt')), \
                            f'No keypoints found for image `{img_path}`.'
                assert len(metadata[video_folder][slice_folder]) > self.center_identity_size, \
                    ''.join([
                        f'Got video slice `{root}/{video_folder}/{slice_folder}` ',
                        f'with number of frames ({len(metadata[video_folder][slice_folder])}) ',
                        f'less than given size of center identity ({self.center_identity_size}).'
                    ])
        return metadata, filelist

    def _generate_center_identities(self):
        cis = list()
        for video_folder in self._videos:
            slice_folders = list(self.metadata_[video_folder].keys())
            random_slice_folder = np.random.choice(slice_folders)
            frames = self.metadata_[video_folder][random_slice_folder]
            random_start_frame_idx = np.random.choice(range(len(frames[:-self.center_identity_size])))
            ci_filelist = self.metadata_[video_folder][random_slice_folder]\
                [random_start_frame_idx:random_start_frame_idx+self.center_identity_size]
            ci = [
                img_as_ubyte(io.imread(f'{self.root}/{video_folder}/{random_slice_folder}/{filename}'))
                for filename in ci_filelist
            ]
            cis.append(ci)
        return cis

    def _normalize(self, x):
        return x * 2 - 1
 
    def __getitem__(self, index):
        if self._cis_counter % self.center_identity_step == 0:
            self._cis = self._generate_center_identities()

        index = index % len(self._videos)

        # Get currect CI
        ci = self._cis[index]
        if self.transform_ci:
            ci = self._normalize(torch.stack([self.transform_ci(image=frame)['image'] for frame in ci]))
        
        # Get random frame from video
        video_folder = self._videos[index]
        slice_folders = list(self.metadata_[video_folder].keys())
        random_slice_folder = np.random.choice(slice_folders)
        random_frame_file = np.random.choice(self.metadata_[video_folder][random_slice_folder])
        frame = img_as_ubyte(io.imread(f'{self.root}/{video_folder}/{random_slice_folder}/{random_frame_file}'))
        keypoints = torch.load(f"{self.root}/{video_folder}/{random_slice_folder}/{random_frame_file.replace('.PNG', '.pt')}")

        # Constructing object
        inputs = {
            'image': frame,
            'keypoints': keypoints.numpy(),
        }
        if self.transform:
            inputs = self.transform(image=inputs['image'], keypoints=inputs['keypoints'])
            inputs['image'] = self._normalize(inputs['image'])
            inputs['keypoints'] = torch.Tensor(inputs['keypoints'])

        # Add transformed CI
        inputs.update({'ci': ci})
        return inputs

    def step(self):
        self._cis_counter += 1
        
    def __len__(self):
        return self.size


class BatchCollate(object):
    def __call__(self, batch):
        batch = {
            'image': torch.stack([item['image'] for item in batch]),
            'keypoints': torch.stack([item['keypoints'] for item in batch]),
            'ci': torch.stack([item['ci'] for item in batch]),
        }
        return batch


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--root", required=True)
    parser.add_argument('-t', "--to_dir", required=True)
    parser.add_argument('-rs', "--resize_shape", default=128, type=int)
    parser.add_argument('-s', "--frame_step", default=2, type=int)
    args = parser.parse_args()

    dataset = VoxCelebDataset(args.root)
    dataset.preprocess_videos(
        to_dir=args.to_dir,
        resize_shape=[args.resize_shape, args.resize_shape],
        frame_step=args.frame_step
    )
    print('Done!')
