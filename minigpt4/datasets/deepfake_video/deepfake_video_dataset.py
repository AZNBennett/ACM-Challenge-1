import os
import json
import cv2
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np

class DeepfakeVideoDataset(Dataset):
    def __init__(self, root_dir, vis_processor, text_processor):
        self.root_dir = root_dir
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.samples = []

        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith(".json"):
                    json_path = os.path.join(dirpath, file)
                    video_path = json_path.replace(".json", ".mp4")
                    if os.path.exists(video_path):
                        self.samples.append((video_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, json_path = self.samples[idx]

        try:
            with open(json_path, 'r') as f:
                meta = json.load(f)
            modify_type = meta.get("modify_type", "real")
            label = 1.0 if modify_type != "real" else 0.0

            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            max_frames = 16
            sample_indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)

            frames = []
            for i in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                success, frame = cap.read()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame)
                processed = self.vis_processor(pil_image)
                frames.append(processed)

            cap.release()

            if not frames:
                return None

            text_prompt = "Given the visual data in this video, determine whether or not the video has been manipulated."

            return {
                "frames": frames,
                "text_input": text_prompt,
                "label": torch.tensor([label], dtype=torch.float)
            }

        except Exception as e:
            print(f"[ERROR] Failed to load {video_path}: {e}")
            return None
