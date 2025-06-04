import os
import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torchvision import transforms

from minigpt4.datasets.deepfake_video.deepfake_video_dataset import DeepfakeVideoDataset
from minigpt4.models.minigpt4 import MiniGPT4
from minigpt4.processors.blip_processors import Blip2ImageEvalProcessor, BlipCaptionProcessor

def safe_collate(batch):
    clean = []
    for sample in batch:
        if sample is None:
            continue
        if (
            isinstance(sample.get("frames"), list) and
            all(isinstance(f, torch.Tensor) for f in sample["frames"]) and
            isinstance(sample.get("label"), torch.Tensor) and
            isinstance(sample.get("text_input"), str) and
            isinstance(sample.get("audio"), torch.Tensor)
        ):
            clean.append(sample)
    return torch.utils.data._utils.collate.default_collate(clean) if clean else None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llama_path = "/root/models/llama2"
    adapter_path = "./output/minigpt4_lora_deepfake_gru/final"
    metadata_path = "./data/val_metadata.json"
    root_dir = "./data/val"

    print("Loading validation dataset...")
    vis_processor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    text_processor = BlipCaptionProcessor()

    dataset = DeepfakeVideoDataset(
        metadata_path=metadata_path,
        root_dir=root_dir,
        vis_processor=vis_processor,
        text_processor=text_processor
    )

    dataloader = DataLoader(dataset, batch_size=1, collate_fn=safe_collate)

    print("Loading model...")
    model_cfg = {
        "llama_model": llama_path,
        "max_txt_len": 160,
        "end_sym": "</s>",
        "low_resource": True,
        "prompt_template": "[INST] {} [/INST]"
    }
    model = MiniGPT4(model_cfg).to(device)
    model.eval()

    if os.path.exists(adapter_path):
        model.llama_model.load_adapter(adapter_path, adapter_name="default")
        print(f"Loaded final LoRA adapter from {adapter_path}")
    else:
        print(f"Final adapter path not found: {adapter_path}")
        return

    criterion = BCEWithLogitsLoss()
    all_labels = []
    all_logits = []
    total_loss = 0
    num_samples = 0

    print("Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            if batch is None:
                continue

            text_input = batch["text_input"][0]
            label = batch["label"].to(device).view(-1)
            audio = batch["audio"].to(device)
            frames = [f.to(device) for f in batch["frames"][0]]

            outputs = model(frames, text_input, audio)
            logit = outputs["logits"].view(-1)
            logit = torch.clamp(logit, min=-20, max=20)

            loss = criterion(logit, label)
            total_loss += loss.item()
            num_samples += 1

            all_labels.append(label.cpu())
            all_logits.append(torch.sigmoid(logit).cpu())

    if not all_labels:
        print("[ERROR] No samples evaluated.")
        return

    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_logits).numpy()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        auc = float('nan')
        print(f"[AUC ERROR] {e}")

    pred_binary = (all_probs >= 0.5).astype(int)
    acc = accuracy_score(all_labels, pred_binary)
    avg_loss = total_loss / num_samples

    print("\nEvaluation Results:")
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
