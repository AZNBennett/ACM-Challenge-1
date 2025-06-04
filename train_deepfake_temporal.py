import os
import gc
import torch
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import AdamW
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import torch.utils.checkpoint
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

torch.utils.checkpoint._use_reentrant = False

from minigpt4.datasets.deepfake_video.deepfake_video_dataset import DeepfakeVideoDataset
from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor, BlipCaptionProcessor

def safe_collate(batch):
    clean = []
    for sample in batch:
        if sample is None:
            print("[SKIP] Sample is None")
            continue
        if (
            isinstance(sample.get("frames"), list) and
            all(isinstance(f, torch.Tensor) for f in sample["frames"]) and
            isinstance(sample.get("label"), torch.Tensor) and
            isinstance(sample.get("text_input"), str) and
            isinstance(sample.get("audio"), torch.Tensor)
        ):
            clean.append(sample)
        else:
            print("[SKIP] Invalid sample:", {k: type(v) for k, v in sample.items()})
    if not clean:
        print("[SKIP] Entire batch filtered out")
        return None
    return default_collate(clean)

def save_checkpoint(save_dir, model):
    ckpt_dir = os.path.join(save_dir, "checkpoint")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.llama_model.save_pretrained(ckpt_dir)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learning_rate = 1e-4

    llama_path = "/root/models/llama2"
    data_path = "./data/train/lrs3/"
    save_dir = "./output/minigpt4_lora_deepfake_gru"
    os.makedirs(save_dir, exist_ok=True)

    print("Loading dataset...")
    import torchvision.transforms as T

    vis_processor = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    text_processor = BlipCaptionProcessor()
    dataset = DeepfakeVideoDataset(
        metadata_path="./data/train_metadata.json",
        root_dir="./data/train",
        vis_processor=vis_processor,
        text_processor=text_processor)
    print(f"Dataset loaded: {len(dataset)} samples")

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

    pos_weight = torch.tensor([1.3096]).to(device)
    criterion = BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    epochs = 15
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

    for epoch in range(epochs):
        all_probs = []
        all_labels = []
        step = 0
        print(f"\nEpoch {epoch + 1}")
        for batch in tqdm(dataloader, desc="Training"):
            if batch is None:
                print("[SKIP] Batch was None")
                continue
            
            text_input = batch["text_input"][0]
            label = batch["label"].to(device).view(-1)
            frames = batch["frames"][0]
            audio = batch["audio"].to(device)
            frames = [f.to(device) for f in frames]

            try:
                outputs = model(frames, text_input, audio)
                video_logit = outputs["logits"]

                
                # AUC logging
                with torch.no_grad():
                    prob = torch.sigmoid(video_logit.detach().view(-1).cpu())
                    label_cpu = label.view(-1).cpu()
                    all_probs.append(prob)
                    all_labels.append(label_cpu)

                if step % 1000 == 0 and len(all_labels) > 1:
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(torch.cat(all_labels).numpy(), torch.cat(all_probs).numpy())
                        print(f"[AUC] Steps {step-99}-{step}: {auc:.4f}")
                    except Exception as e:
                        print(f"[AUC ERROR] {e}")
                    all_probs.clear()
                    all_labels.clear()
                

                if torch.isnan(video_logit).any() or torch.isinf(video_logit).any():
                    print(f"[NaN/Inf] Logit at step {step}: {video_logit}")
                    step += 1
                    continue

                video_logit = torch.clamp(video_logit, min=-20, max=20)
                loss = criterion(video_logit.view(-1), label.view(-1))

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"[NaN] Loss at step {step}: {loss.item()}")
                    step += 1
                    continue

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                tqdm.write(f"Step {step}, Loss: {loss.item():.4f}")

                if step % 1000 == 0:
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
