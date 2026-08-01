# Retinal_Vessel_Segmentation_UNet
Automated Retinal Vessel Segmentation using PyTorch U-Net, CLAHE Preprocessing, and Custom Dice-Focal Loss on the DRIVE Dataset
🌐 **Live Web Application:** [Try the Retinal Vessel Segmentation App Live on Hugging Face](https://huggingface.co/spaces/SaiGanesh-26/retinal-vessel-segmentation)
![Retinal Vessel Segmentation Pipeline](retinal%20vessel%20segmentation.png)
*Figure: (1) Original RGB Fundus Scan, (2) Green Channel + CLAHE Contrast Enhancement, (3) U-Net Predicted Binary Mask, (4) Semi-Transparent Red Vessel Overlay.*



## Key Metrics & Highlights

* **Model Architecture:** Custom U-Net built from scratch in PyTorch
* **Loss Function:** Hybrid Dice-Focal Loss (designed for 90:10 background-to-vessel class imbalance)
* **Dataset:** DRIVE (Digital Retinal Images for Vessel Extraction)
* **Validation Performance:**
  * **Dice Score / F1-Score:** ~0.80+
  * **Sensitivity (Recall):** ~0.78+
  * **Specificity:** ~0.97+



## Pipeline Architecture

1. **Preprocessing:** Green-channel extraction + Contrast Limited Adaptive Histogram Equalization (CLAHE) to sharpen fine micro-capillary boundaries.
2. **Augmentation:** Synchronized Elastic Transforms, Rotations, and Flips via Albumentations to prevent overfitting on small medical datasets.
3. **Model:** Contracting encoder path with `MaxPool2d` and expanding decoder path with `ConvTranspose2d` and skip connections (`torch.cat`).
4. **Post-Processing:** Morphological Opening and Closing via OpenCV to eliminate isolated background noise and bridge thin vessel breaks.

---

##  Quickstart & Execution

```bash
# Clone Repository
git clone [https://github.com/](https://github.com/)<YOUR_GITHUB_USERNAME>/retinal-vessel-segmentation-unet.git
cd retinal-vessel-segmentation-unet

# Install Requirements
pip install -r requirements.txt
```

Run the complete pipeline directly in Google Colab using `notebook/retinal_vessel_segmentation.ipynb`.
