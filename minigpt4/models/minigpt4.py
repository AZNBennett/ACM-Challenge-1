import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

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

        self.hidden_dim = self.llama_model.config.hidden_size

        self.gru = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
        )

        self.classifier = nn.Linear(self.hidden_dim, 1)

    def encode_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len
        )
        return inputs.input_ids.cuda(), inputs.attention_mask.cuda()

    def forward(self, frames, text_input):
        input_ids, attention_mask = self.encode_text(text_input)
        frame_embeddings = []

        for i, image in enumerate(frames):
            image = image.unsqueeze(0).cuda()

            try:
                with autocast():
                    outputs = self.llama_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        return_dict=True
                    )

                cls_embed = outputs.hidden_states[-1][:, 0, :]
                if torch.isnan(cls_embed).any() or torch.isinf(cls_embed).any():
                    print(f"[WARN] NaN in CLS embed for frame {i}, skipping")
                    continue

                frame_embeddings.append(cls_embed.squeeze(0))

            except Exception as e:
                print(f"[ERROR] Frame {i} forward pass failed: {e}")
                continue

        if not frame_embeddings:
            print("[SKIP] No usable frames in video — returning dummy logit")
            return {"logits": torch.tensor(0.0).cuda().requires_grad_()}

        seq = torch.stack(frame_embeddings).unsqueeze(0)  # [1, T, D]
        _, h_n = self.gru(seq)

        if torch.isnan(h_n).any() or torch.isinf(h_n).any():
            print("[MODEL ERROR] NaN/Inf in GRU hidden state")
            return {"logits": torch.tensor(0.0).cuda().requires_grad_()}

        logits = self.classifier(h_n.squeeze(0))
        return {"logits": logits.squeeze(0)}
