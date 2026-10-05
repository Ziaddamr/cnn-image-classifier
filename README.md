# 🏛 Indoor Scene Style Classification (EVA-02 Vision Transformer)

Hey there! 👋 This repository contains a deep learning pipeline built in PyTorch that classifies indoor scene styles with high accuracy. Instead of a standard CNN, I used a fine-tuned EVA-02 Large Vision Transformer (`eva02_large_patch14_224`) via the `timm` library to leverage state-of-the-art vision models.

## 💡 What Makes This Model Work Well?

Here are the key design choices I made while building and training this network:

* **Stochastic Depth:** Set `drop_path_rate=0.3` on the EVA-02 Large backbone so the model doesn't over-rely on specific attention layers during training.
* **Layer-Wise Learning Rate Decay (LLRD):** Grouped the parameters with differential learning rates to preserve pretrained lower-level features while aggressively training the classification head:
  * Patch Embeddings: 1 × 10⁻⁶
  * Transformer Blocks: 1 × 10⁻⁵
  * Classification Head: 4 × 10⁻⁴
* **Heavy Data Augmentation:** Used Bicubic interpolation to keep image details sharp, combined with `TrivialAugmentWide`, `RandomResizedCrop`, and `RandomErasing (p=0.3)` to prevent overfitting.
* **Optimization Setup:**
  * **Mixed Precision Training:** Used PyTorch `amp.autocast` and `GradScaler` to cut memory usage and boost GPU speed.
  * **Gradient Accumulation:** Accumulated gradients over mini-batches to reach an effective batch size of 32.
  * **Loss & Scheduling:** Applied `CrossEntropyLoss` with Label Smoothing (0.1) to handle boundary cases better, paired with a `CosineAnnealingLR` scheduler over 16 epochs using AdamW.
* **Reliable Inference:** Used a stratified train/validation split and a clean inference loop that automatically outputs formatted test set predictions (`submission.csv`).

## 🛠 Tech Stack

* **Language:** Python
* **Deep Learning:** PyTorch, torchvision, timm
* **Data & Utilities:** Pandas, NumPy, PIL, scikit-learn, tqdm
