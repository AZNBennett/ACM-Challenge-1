import os
import gc
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import torch.utils.checkpoint

torch.utils.checkpoint._use_reentrant = False

from minigpt4.datasets.deepfake_video.deepfake_video_dataset import DeepfakeVideoDataset
from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor, BlipCaptionProcessor
# from torch.cuda.amp import GradScaler, autocast  # Optional for AMP stability

def safe_collate(batch):
    clean = []
    for sample in batch:
        if sample is None:
            continue
        if (
            isinstance(sample.get("frames"), list) and
            all(isinstance(f, torch.Tensor) for f in sample["frames"]) and
            isinstance(sample.get("label"), torch.Tensor) and
            isinstance(sample.get("text_input"), str)
        ):
            clean.append(sample)
    if not clean:
        return None
    return default_collate(clean)

def save_checkpoint(save_dir, model):
    ckpt_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.llama_model.save_pretrained(ckpt_dir)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    llama_path = "/root/models/llama2"
    data_path = "./data/train/lrs3/"
    save_dir = "./output/minigpt4_lora_deepfake_gru"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...")
    vis_processor = Blip2ImageEvalProcessor(image_size=224)
    text_processor = BlipCaptionProcessor()
    dataset = DeepfakeVideoDataset(data_path, vis_processor, text_processor)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=safe_collate,
        num_workers=0
    )

    print("Loading model...")
    model_cfg = {
        "llama_model": llama_path,
        "max_txt_len": 160,
        "end_sym": "</s>",
        "low_resource": True,
        "prompt_template": "[INST] {} [/INST]"
    }
    model = MiniGPT4(model_cfg).to(device)
    model.train()

    adapter_path = os.path.join(save_dir, "checkpoint")
    if os.path.exists(adapter_path) and os.path.isfile(os.path.join(adapter_path, "adapter_model.safetensors")):
        model.llama_model.load_adapter(adapter_path, adapter_name="default")
        print("Loaded LoRA adapter from /checkpoint/")

    criterion = BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # scaler = GradScaler()  # Optional

    step = 0
    for epoch in range(1):
        print(f"\nEpoch {epoch + 1}")
        for batch in tqdm(dataloader, desc="Training"):
            if batch is None:
                continue

            text_input = batch["text_input"][0]
            label = batch["label"].to(device).view(-1)
            frames = batch["frames"][0]

            try:
                outputs = model(frames, text_input)
                video_logit = outputs["logits"]

                # Diagnostic: check for bad logits
                if torch.isnan(video_logit).any() or torch.isinf(video_logit).any():
                    print(f"[NaN/Inf] Logit at step {step}: {video_logit}")
                    step += 1
                    continue

                # Clamp logits to avoid extreme BCE penalty
                video_logit = torch.clamp(video_logit, min=-20, max=20)

                loss = criterion(video_logit.view(-1), label.view(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[NaN/Inf] Loss at step {step}: {loss.item()}")
                    step += 1
                    continue

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # scaler.scale(loss).backward()
                # scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                # scaler.step(optimizer)
                # scaler.update()

                optimizer.step()

                tqdm.write(f"Step {step}, Loss: {loss.item():.4f}")

                if step % 100 == 0:
                    save_checkpoint(save_dir, model)

            except Exception as e:
                print(f"[ERROR] Exception at step {step}: {e}")
                step += 1
                continue

            del outputs, video_logit, loss, frames
            torch.cuda.empty_cache()
            gc.collect()

            step += 1

    final_path = os.path.join(save_dir, "final")
    model.llama_model.save_pretrained(final_path)
    print(f"\nLoRA adapters saved to {final_path}")

if __name__ == "__main__":
    main()
