# Trilingual Image Captioning (Bengali–Hindi–English)

A trilingual image captioning system that generates captions in **Bengali**, **Hindi**, and **English** from a single image using a shared frozen **CLIP ViT-L/14** visual encoder, a lightweight **MLP prefix mapper**, and three **language-specific GPT-2 decoders** (BanglaGPT, GPT2-Hindi, and GPT-2), following the CLIPCap prefix-tuning paradigm.

This repository accompanies the paper *"Towards Trilingual Image Captioning: Bridging Bengali, Hindi, and English Through Shared Visual Encoding and Language-Native Text Generation."*

---

## Overview

- **Visual Encoder:** Frozen CLIP ViT-L/14 (768-dim CLS embedding), shared across all three languages
- **MLP Mapper:** Linear(768→2048) → Tanh → Linear(2048→10×768), producing 10 prefix tokens
- **Decoders:**
  - Bengali → BanglaGPT (GPT-2, vocab 50,002)
  - Hindi → GPT2-Hindi (GPT-2, vocab 50,258)
  - English → GPT-2 (vocab 50,257)
- **Datasets:** BNATURE (Bengali), PASCAL 1K-Hindi (Hindi), Flickr8k (English), each split 80/10/10 at the image level

## Results Summary

| Metric | Bengali | Hindi | English |
|---|---|---|---|
| BLEU-4 (NLTK) | 18.00 | 11.27 | 18.51 |
| CIDEr | 57.28 | 35.63 | 40.01 |
| METEOR | 38.94 | 26.53 | 46.41 |
| ROUGE-L | 63.47 | 47.60 | 43.50 |
| BERTScore-F1 | 81.09 | 74.76 | 90.40 |

Full results, ablations, and analysis are reported in the paper.

## Repository Structure
├── app.py # UI/entry point for running captioning on an image
├── model.py # CLIPCap architecture: MLP mapper + GPT-2 decoder wrapper
├── inference.py # Caption generation logic (greedy / beam search)
├── requirements.txt # Python dependencies
├── checkpoints/ # Trained model weights (NOT included — see below)
│ ├── clipcap_bn_best.pt
│ ├── clipcap_hi_best.pt
│ └── clipcap_en_best.pt
└── README.md
## Model Checkpoints

Due to file size limits, trained model checkpoints (`.pt` files) are **not included** in this repository.

They are available at: https://www.kaggle.com/code/gppppp23/image-caption-generation-with-clip-transformer15

Download the checkpoints and place them inside a local `checkpoints/` folder before running inference, matching the paths defined in `app.py`:

```python
CKPT_PATHS = {
    "bn": "checkpoints/clipcap_bn_best.pt",
    "hi": "checkpoints/clipcap_hi_best.pt",
    "en": "checkpoints/clipcap_en_best.pt",
}
```

## Installation

```bash
git clone https://github.com/Bijoy89/Trilingual-Image-Captioning.git
cd Trilingual-Image-Captioning
pip install -r requirements.txt
```

## Usage

1. Download the checkpoints (see above) into `checkpoints/`.
2. Run the app:

```bash
python app.py
```

3. Select an image and a target language (Bengali / Hindi / English) to generate a caption.

Alternatively, use `inference.py` directly for programmatic caption generation:

```python
from inference import generate_caption

caption = generate_caption(image_path="example.jpg", lang="bn")
print(caption)
```

## Training

Training was performed on Kaggle using cached CLIP ViT-L/14 embeddings for all dataset images. The full training notebook is publicly available at:
https://www.kaggle.com/code/gppppp23/image-caption-generation-with-clip-transformer15

Key training settings:

| Language | Epochs | Beam (default) |
|---|---|---|
| Bengali | 10 | 5 |
| Hindi | 14 | 5 |
| English | 10 | 5 |

## Datasets

- **BNATURE** (Bengali) — [Al-Faraby et al.]
- **PASCAL 1K-Hindi** (Hindi) — [Sharma et al., ChitraVivran]
- **Flickr8k** (English)

No new datasets were created for this work; all datasets are publicly available and cited in the paper.

## Citation

If you use this code, please cite:

```bibtex
@article{bhattacharjee2026trilingual,
  title={Towards Trilingual Image Captioning: Bridging Bengali, Hindi, and English Through Shared Visual Encoding and Language-Native Text Generation},
  author={Bhattacharjee, Bijoy and Al Ahsan, Md. Sabbir},
  journal={The Visual Computer},
  year={2026}
}
```

## License
MIT

