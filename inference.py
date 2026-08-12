# inference.py
import torch
import open_clip
from PIL import Image
import torchvision.transforms as T
from model import CLIPCapModel, DEVICE

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

TFM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(CLIP_MEAN, CLIP_STD),
])

_clip_model = None


def _load_clip():
    global _clip_model
    if _clip_model is None:
        print("Loading CLIP ViT-L/14 ...")
        model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model = model.to(DEVICE).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _clip_model = model
        print("CLIP ready.")
    return _clip_model


def extract_clip_embedding(image_path: str) -> torch.Tensor:
    clip = _load_clip()
    img = Image.open(image_path).convert("RGB")
    tensor = TFM(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = clip.encode_image(tensor).cpu().float().squeeze(0)
    return emb


_caption_models: dict[str, CLIPCapModel] = {}


def load_caption_model(lang: str, ckpt_path: str) -> CLIPCapModel:
    if lang not in _caption_models:
        print(f"Loading caption model [{lang}] ...")
        model = CLIPCapModel(lang).to(DEVICE)
        model.load_weights(ckpt_path)
        model.eval()
        _caption_models[lang] = model
        print(f"Caption model [{lang}] ready.")
    return _caption_models[lang]


def generate_caption(
    image_path: str,
    lang: str,
    ckpt_path: str,
    beam: int = 5,
) -> str:
    emb   = extract_clip_embedding(image_path)
    model = load_caption_model(lang, ckpt_path)
    return model.generate(emb, beam=beam)