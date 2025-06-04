import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig
import torchvision.models as models
import torchvision.transforms as T

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel

@registry.register_model("minigpt4_deepfake")
class MiniGPT4(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}

    def __init__(self, model_cfg, *args, **kwargs):
        super().__init__()

        llama_model = model_cfg.get("llama_model")
        self.max_txt_len = model_cfg.get("max_txt_len", 160)
        self.low_resource = model_cfg.get("low_resource", False)
        self.end_sym = model_cfg.get("end_sym", "</s>")
        self.prompt_template = model_cfg.get("prompt_template", '[INST] {} [/INST] ')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            llama_model,
            quantization_config=bnb_config,
            device_map="auto",
        )
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)

        lora_config = LoraConfig(
            r=64,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama_model = get_peft_model(self.llama_model, lora_config)

        self.tokenizer = AutoTokenizer.from_pretrained(
            llama_model,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        from torchvision.models import resnet34, ResNet34_Weights
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT)

        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.vision_encoder.eval()
        self.vision_encoder.requires_grad_(False)

        self.hidden_dim = 512

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.audio_proj = nn.Linear(64, self.hidden_dim)
        self.fusion_fc = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.dropout = nn.Dropout(0.2)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )


        self.resnet_norm = T.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])

    def encode_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len
        )
        return inputs.input_ids.cuda(), inputs.attention_mask.cuda()

    def forward(self, frames, text_input, audio):
        input_ids, attention_mask = self.encode_text(text_input)
        frame_embeddings = []

        for i, image in enumerate(frames):
            image = image.unsqueeze(0).cuda()

            try:
                image = self.resnet_norm(image)
                with torch.no_grad():
                    feat = self.vision_encoder(image)
                    feat = feat.view(1, -1)
                frame_embeddings.append(feat.squeeze(0))

            except Exception as e:
                print(f"[ERROR] Frame {i} vision encoding failed: {e}")
                continue

        if not frame_embeddings:
            print("[SKIP] No usable frames in video — returning dummy logit")
            return {"logits": torch.tensor(0.0).cuda().requires_grad_()}

        seq = torch.stack(frame_embeddings).unsqueeze(0)
        _, h_n = self.gru(seq)

        if torch.isnan(h_n).any() or torch.isinf(h_n).any():
            print("[MODEL ERROR] NaN/Inf in GRU hidden state")
            return {"logits": torch.tensor(0.0).cuda().requires_grad_()}

        visual_feat = h_n.squeeze(0)
        audio = audio.to(visual_feat.device)

        try:
            audio_feat = torch.mean(audio, dim=-1)
            audio_feat = self.audio_proj(audio_feat)
        except Exception as e:
            print(f"[WARN] Audio processing failed: {e}")
            audio_feat = torch.zeros_like(visual_feat)

        visual_feat = F.normalize(visual_feat, dim=-1)
        audio_feat = F.normalize(audio_feat, dim=-1)

        combined = torch.cat([visual_feat, audio_feat], dim=-1)
        combined = self.fusion_fc(combined)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        return {"logits": logits.squeeze(0)}
