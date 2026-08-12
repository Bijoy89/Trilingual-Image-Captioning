# model.py
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DECODER_MAP = {
    "bn": "shahidul034/BanglaGPT",
    "hi": "surajp/gpt2-hindi",
    "en": "gpt2",
}

GEN_CONFIG = {
    "bn": {"max_new_tokens": 50, "no_repeat_ngram_size": 0, "repetition_penalty": 1.3},
    "hi": {"max_new_tokens": 50, "no_repeat_ngram_size": 0, "repetition_penalty": 1.3},
    "en": {"max_new_tokens": 40, "no_repeat_ngram_size": 3, "repetition_penalty": 1.2},
}


def _clean_caption(text: str, lang: str) -> str:
    if not text or not text.strip():
        return "."
    text = text.strip().split("\n")[0].strip()
    if lang == "bn":
        parts = [p.strip() for p in re.split(r"[।!?]", text) if p.strip()]
        for part in parts:
            if len(part.split()) >= 4:
                return part.strip() + "।"
        words = text.split()
        return " ".join(words[:15]) + "।" if len(words) >= 4 else "."
    elif lang == "hi":
        parts = [p.strip() for p in re.split(r"[।!?]", text) if p.strip()]
        for part in parts:
            if len(part.split()) >= 4:
                return part.strip() + "।"
        words = text.split()
        return " ".join(words[:15]) + "।" if len(words) >= 4 else "."
    else:
        m = re.match(r'^([^.!?]+[.!?])', text)
        if m:
            words = m.group(1).split()
            if 4 <= len(words) <= 22:
                return m.group(1).strip()
        words = text.split()
        if len(words) > 20:
            return " ".join(words[:20]) + "."
        return text.rstrip(".,") + "." if len(text.split()) >= 4 else "."


class MLP(nn.Module):
    def __init__(self, clip_dim: int, gpt2_dim: int, prefix_len: int, hidden: int = 2048):
        super().__init__()
        self.prefix_len = prefix_len
        self.fc1 = nn.Linear(clip_dim, hidden)
        self.act = nn.Tanh()
        self.fc2 = nn.Linear(hidden, prefix_len * gpt2_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))).view(x.size(0), self.prefix_len, -1)


class CLIPCapModel(nn.Module):
    def __init__(self, lang: str, clip_dim: int = 768, prefix_len: int = 10):
        super().__init__()
        self.lang = lang
        self.prefix_len = prefix_len

        model_id = DECODER_MAP[lang]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self.gpt2.resize_token_embeddings(len(self.tokenizer))

        gpt2_dim = self.gpt2.config.hidden_size
        self.mapper = MLP(clip_dim, gpt2_dim, prefix_len)

    def load_weights(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.mapper.load_state_dict(ck["mapper"])
        self.gpt2.load_state_dict(ck["gpt2"])
        print(f"Loaded: {ckpt_path}")

    @torch.no_grad()
    def generate(self, clip_emb: torch.Tensor, beam: int = 5) -> str:
        self.eval()
        if clip_emb.dim() == 1:
            clip_emb = clip_emb.unsqueeze(0)
        clip_emb = clip_emb.to(DEVICE).float()
        prefix = self.mapper(clip_emb)
        attn_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=DEVICE)
        cfg = GEN_CONFIG[self.lang]
        out = self.gpt2.generate(
            inputs_embeds=prefix,
            attention_mask=attn_mask,
            max_new_tokens=cfg["max_new_tokens"],
            min_new_tokens=5,
            num_beams=beam,
            no_repeat_ngram_size=cfg["no_repeat_ngram_size"],
            repetition_penalty=cfg["repetition_penalty"],
            length_penalty=1.2,
            early_stopping=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
        return _clean_caption(text, self.lang)