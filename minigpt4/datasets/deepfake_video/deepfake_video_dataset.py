import os
import json
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
import numpy as np
import imageio.v3 as iio
import torchaudio.transforms as T
import av

class DeepfakeVideoDataset(Dataset):
    def __init__(self, metadata_path, root_dir, vis_processor, text_processor):
        self.root_dir = root_dir
        self.vis_processor = vis_processor
        self.text_processor = text_processor
        self.samples = []

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        for entry in metadata:
            modify_type = entry.get("modify_type", "real")
            video_path = os.path.join(root_dir, entry["file"])
            self.samples.append({"video_path": video_path, "modify_type": modify_type})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample["video_path"]
        modify_type = sample["modify_type"]
        label = 0.0 if modify_type == "real" else 1.0

        try:
            all_frames = list(iio.imiter(video_path))
            total_frames = len(all_frames)
            if total_frames == 0:
                raise ValueError("No frames decoded")
            
            # Sampling 32 Consecutive Frames

            max_frames = 32
            if total_frames > max_frames:
                start = np.random.randint(0, total_frames - max_frames + 1)
                sample_indices = list(range(start, start + max_frames))
            else:
                sample_indices = list(range(total_frames))

            frames = []
            for i in sample_indices:
                frame = all_frames[i]
                pil_image = Image.fromarray(frame)
                processed = self.vis_processor(pil_image)
                frames.append(processed)

        except Exception as e:
            print(f"[FAIL] Frame extraction failed for: {video_path} | {e}")
            return None

        if not frames:
            print(f"[FAIL] No valid frames in: {video_path}")
            return None

        text_prompt = "Given the visual data in this video, determine whether or not the video has been manipulated."

        try:
            container = av.open(video_path)
            audio_stream = next(s for s in container.streams if s.type == 'audio')
            resampler = av.audio.resampler.AudioResampler(format='s16', layout='mono', rate=16000)

            waveform = []
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                if isinstance(resampled, list):
                    for subframe in resampled:
                        waveform.append(subframe.to_ndarray())
                else:
                    waveform.append(resampled.to_ndarray())

            if not waveform:
                raise ValueError("No audio decoded")

            audio_np = np.concatenate(waveform, axis=1).squeeze()
            audio_tensor = torch.tensor(audio_np, dtype=torch.float32) / 32768.0

            audio_transform = T.MelSpectrogram(sample_rate=16000, n_mels=64)
            mel_spec = audio_transform(audio_tensor.unsqueeze(0)).squeeze(0)
            mel_spec = torch.clamp(mel_spec, min=-1.0, max=1.0)

        except Exception as e:
            print(f"[WARN] PyAV failed to load audio from {video_path}: {e}")
            mel_spec = torch.zeros((64, 100))

        return {
            "frames": frames,
            "audio": mel_spec,
            "text_input": text_prompt,
            "label": torch.tensor([label], dtype=torch.float)
        }
