import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, get_peft_model, LoraConfig

from minigpt4.common.registry import registry
from minigpt4.models.base_model import BaseModel


@registry.register_model("minigpt4")
class MiniGPT4(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {}  # Required for MiniGPT-4 registry

    def __init__(self, model_cfg, *args, **kwargs):
        super().__init__(model_cfg, *args, **kwargs)

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

        self.llama_model.gradient_checkpointing_enable()
        self.llama_model = prepare_model_for_kbit_training(self.llama_model)

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.llama_model = get_peft_model(self.llama_model, lora_config)

        print("🔧 Trainable parameters (LoRA):")
        for name, param in self.llama_model.named_parameters():
            if param.requires_grad:
                print(" -", name)

        self.tokenizer = AutoTokenizer.from_pretrained(
            llama_model,
            use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token

        self.classifier = nn.Sequential(
            nn.Linear(self.llama_model.config.hidden_size, 1)
        )

    def encode_text(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len
        )
        return inputs.input_ids.cuda(), inputs.attention_mask.cuda()

    def forward(self, samples):
        image = samples["image"].unsqueeze(0).cuda()
        text = samples["text_input"]
        input_ids, attention_mask = self.encode_text(text)

        with autocast():
            outputs = self.llama_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )

            last_hidden = outputs.hidden_states[-1]        # [B, T, D]
            cls_embedding = last_hidden[:, 0, :]            # [B, D]

            logits = self.classifier(cls_embedding).squeeze(1)  # [B]

        return {
            "logits": logits,
            "raw_outputs": outputs
        }
