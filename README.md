# web-ui-code-generation
Code generation from web UI images using deep learning models inspired by image captioning techniques (CNN, ViT, RNN, Attention)


# Web UI Code Generation

This project aims to generate HTML code from web UI screenshots using deep learning models.  
Inspired by image captioning architectures, it combines CNNs, attention mechanisms, and Vision Transformers to produce structured HTML descriptions from images.

## 🧠 Inspiration

Based on: [image-captioning by JeremyNathanJusuf](https://github.com/JeremyNathanJusuf/image-captioning)

## 📂 Project Overview

- Preprocess GUI images
- Encode images using CNN or ViT
- Decode with RNN / Attention to generate token sequences representing HTML
- Evaluate with BLEU score and qualitative samples

## 🧪 Models Tested

| Model           | Architecture                | BLEU Score |
|----------------|-----------------------------|------------|
| CNN-RNN        | CNN + RNN                   | 72.47%     |
| CNN-Attn       | CNN + Attention             | 60.88%     |
| ViT-Attn       | Vision Transformer + Attn   | 61.22%     |
| YOLO-Attn      | YOLO + Attention            | 57.62%     |
| YOLOCNN-Attn   | YOLO + CNN + Attention      | 62.74%     |
| ViTCNN-Attn    | ViT + CNN + Attention       | 46.07%     |

## 🖼️ Example Results

Examples of generated HTML components from GUI images are included in the `results/` folder.

## ⚙️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt


 Run Training
bash
Copier
Modifier
python src/train.py




📍 Future Work
Add DOM-tree-aware generation (structural constraints)

Guide generation with component segmentation maps

Explore multimodal conditioning for better results

📄 License
MIT (or adapt depending on your reuse policy)

yaml
Copier
Modifier

---

### 3. 📜 `requirements.txt`

Exemple basique si tu gardes le même environnement que le repo original :

```txt
torch>=1.10
torchvision
numpy
Pillow
tqdm
matplotlib
scikit-learn
opencv-python
transformers
Ajuste selon tes besoins.

4. 🧹 .gitignore (important)
txt
Copier
Modifier
*.pyc
__pycache__/
.ipynb_checkpoints/
models/
*.pt
*.h5
*.ckpt
