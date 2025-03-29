import os
import json
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from config import *

class FusionDataset(Dataset):
    def __init__(self, split='train', num_frames=NUM_FRAMES):
        self.split = split
        self.num_frames = num_frames
        self.rgb_dir, self.skeleton_dir = self._get_split_paths()
        self.clip_ids = []
        self.data = self._build_dataset()
        self.rgb_transform = self._build_rgb_transform()
        
        if split == 'train':
            self.skeleton_mean, self.skeleton_std = self._compute_skeleton_stats()
        else:
            self.skeleton_mean = None
            self.skeleton_std = None

    def _get_split_paths(self):
        if self.split == 'train':
            return ROOT_DIR, SKELETON_ROOT
        elif self.split == 'val':
            return VAL_DIR, VAL_SKELETON_ROOT
        elif self.split == 'test':
            return TEST_DIR, TEST_SKELETON_ROOT
        else:
            raise ValueError(f"Invalid split: {self.split}. Use 'train' or 'val'.")

    def _build_dataset(self):
        data = []
        skipped_frames = 0
        invalid_clips = 0
        
        for class_name in os.listdir(self.rgb_dir):
            class_dir = os.path.join(self.rgb_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            image_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
            clip_ids = {f.split('.')[0] for f in image_files}
            
            for clip_id in clip_ids:
                clip_frames = []
                #valid_frames = 0
                has_valid_frames = False
                
                for f in image_files:
                    if not f.startswith(f"{clip_id}."):
                        continue

                    rgb_path = os.path.join(class_dir, f)
                    base_name = os.path.splitext(f)[0]
                    skeleton_path = os.path.join(
                        self.skeleton_dir, class_name, "json",
                        f"{base_name}_keypoints.json"
                    )

                    # Always add the frame, mark invalid skeletons
                    if self._is_valid_skeleton(skeleton_path):
                        clip_frames.append((rgb_path, skeleton_path))
                        has_valid_frames = True
                    else:
                        clip_frames.append((rgb_path, None))
                        skipped_frames += 1
                    """ if not self._is_valid_skeleton(skeleton_path):
                        skipped_frames += 1
                        # Keep frame but mark as invalid
                        clip_frames.append((rgb_path, None))  # None indicates invalid skeleton
                    else:        
                        clip_frames.append((rgb_path, skeleton_path))
                        has_valid_frames = True
                        #valid_frames += 1 """

                #if valid_frames >= 1:
                # Always add the clip even if all skeletons are invalid
                data.append((clip_frames, int(class_name)-1))
                self.clip_ids.append(clip_id)
                if not has_valid_frames:
                    invalid_clips += 1

        print(f"{self.split.capitalize()} Dataset:")
        print(f"- Clips: {len(data)}")
        print(f"- Skipped frames: {skipped_frames}")
        print(f"- Clips with all invalid frames: {invalid_clips}")
        return data

    def _is_valid_skeleton(self, path):
        if not os.path.exists(path) or os.path.getsize(path) < 10:
            return False
        try:
            with open(path) as f:
                data = json.load(f)
            return bool(data.get('people') and data['people'][0].get('pose_keypoints_2d'))
        except Exception:
            return False

    def _build_rgb_transform(self):
        if self.split == 'train':
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        clip_frames, label = self.data[idx]
        frame_indices = self._sample_frames(len(clip_frames))
        
        rgb_frames = []
        skeleton_frames = []
        
        for i in frame_indices:
            rgb_path, skeleton_path = clip_frames[i]
            
            # Load RGB
            img = Image.open(rgb_path).convert('RGB')
            img = self.rgb_transform(img)
            
            # ALWAYS load through standard path
            raw_kps = self._load_keypoints(skeleton_path)
            valid_skeleton = np.any(~np.isnan(raw_kps))  # Check validity
            
            # Normalization
            if self.skeleton_mean is not None and self.skeleton_std is not None:
                valid_mask = ~np.isnan(raw_kps)
                normalized_kps = np.zeros_like(raw_kps)  # Initialize with zeros
                
                # Only normalize valid keypoints
                normalized_kps[valid_mask] = (
                    raw_kps[valid_mask] - self.skeleton_mean[valid_mask]
                ) / (self.skeleton_std[valid_mask] + 1e-6)
            else:
                normalized_kps = raw_kps

            # Force invalid confidence to 0 AFTER normalization
            if not valid_skeleton:
                normalized_kps[2::3] = 0  # Zero confidence scores
            else:
                # Still check individual joint confidence
                conf_mask = normalized_kps[2::3] < 0.1
                normalized_kps[2::3][conf_mask] = 0

            skeleton_frames.append(torch.from_numpy(normalized_kps).float())
            rgb_frames.append(img)
        
        return (
            torch.stack(rgb_frames).permute(1, 0, 2, 3),
            torch.stack(skeleton_frames),
            label
        )
    
    def _load_keypoints(self, path):
        """Handle missing/invalid paths by returning NaNs"""
        if not path or not self._is_valid_skeleton(path):
            return np.full(75, np.nan, dtype=np.float32)
        
        try:
            with open(path) as f:
                data = json.load(f)
            kps = np.array(data['people'][0]['pose_keypoints_2d'], dtype=np.float32)
            
            # Mark low-confidence points as invalid
            kps[2::3] = np.where(kps[2::3] < 0.1, np.nan, kps[2::3])
            return kps
        except Exception:
            return np.full(75, np.nan, dtype=np.float32)

    """ def _load_keypoints(self, path):
        default_kps = np.zeros(25*3, dtype=np.float32)
        try:
            # Validate path type before processing
            if not isinstance(path, (str, bytes, os.PathLike)):
                return default_kps
                
            if not os.path.exists(path) or os.path.getsize(path) < 10:
                return default_kps
            
            if not path or not self._is_valid_skeleton(path):
                return np.full(75, np.nan, dtype=np.float32) # Use NaN for invalid
                
            with open(path) as f:
                data = json.load(f)
                
            if not data.get('people') or len(data['people']) == 0:
                return default_kps
                
            kps = data['people'][0].get('pose_keypoints_2d')
            # Zero out low-confidence points
            kps[2::3] = np.where(kps[2::3] < 0.1, np.nan, kps[2::3])

            if not kps or len(kps) != 75:
                return default_kps
                
            return np.array(kps, dtype=np.float32)
            
        except Exception as e:
            print(f"Skeleton load failed: {path} - {str(e)}")
            return default_kps
 """
    def _sample_frames(self, total_frames):
        if total_frames >= self.num_frames:
            start = random.randint(0, total_frames - self.num_frames)
            return list(range(start, start + self.num_frames))
        else:
            return (np.arange(total_frames).tolist() * (self.num_frames // total_frames + 1))[:self.num_frames]

    """ def _compute_skeleton_stats(self):
        all_keypoints = []
        for clip_frames, _ in self.data:
            for _, skeleton_path in clip_frames:
                if skeleton_path and self._is_valid_skeleton(skeleton_path):
                    kps = self._load_keypoints(skeleton_path)
                    all_keypoints.append(kps)  # Keep all 75 values
        
        if not all_keypoints:
            return np.zeros(75), np.ones(75)
        
        return np.nanmean(all_keypoints, axis=0), np.nanstd(all_keypoints, axis=0) """
    def _compute_skeleton_stats(self):
        """Compute normalization statistics"""
        all_keypoints = []
        for clip_frames, _ in self.data:
            for frame in clip_frames:  # Each frame is (rgb_path, skeleton_path)
                skeleton_path = frame[1]  # Get the skeleton path from tuple
                if skeleton_path and self._is_valid_skeleton(skeleton_path):
                    kps = self._load_keypoints(skeleton_path)
                    all_keypoints.append(kps)
        
        if not all_keypoints:
            return np.zeros(75), np.ones(75)
        
        all_keypoints = np.array(all_keypoints)
        mean = np.zeros(75)
        std = np.ones(75)
        
        for i in range(75):
            feature_values = all_keypoints[:, i]
            valid_values = feature_values[~np.isnan(feature_values)]
            
            if len(valid_values) > 0:
                mean[i] = np.mean(valid_values)
                std[i] = np.std(valid_values)
                if std[i] < 1e-6:
                    std[i] = 1.0
        
        return mean, std

def get_dataloaders():
    train_dataset = FusionDataset(split='train')
    val_dataset = FusionDataset(split='val')

    val_dataset.skeleton_mean = train_dataset.skeleton_mean
    val_dataset.skeleton_std = train_dataset.skeleton_std

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader