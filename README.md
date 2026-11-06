# PharmaVision — mobile tablet screening (research demo)

A minimal, responsive browser application for **your independently trained EfficientNetV2-S** model. It is a research prototype, **not** the original authors' software or a validated pharmaceutical inspection system.

- **Frontend:** plain HTML, CSS, JavaScript (no React, no Node build)
- **Inference:** ONNX Runtime Web, **WebAssembly CPU** in the browser
- **Input:** one tablet photograph, manually cropped, 224 × 224 RGB
- **Preprocessing:** RGB values divided by 255; ImageNet normalization `mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`
- **Output:** two logits, class 0 = `defective`, class 1 = `normal`; softmax with **defective threshold 0.40**
- **Privacy:** camera/photo stays in the user's browser during inference; the browser downloads the ONNX runtime from jsDelivr and the model from the website's host; no image-upload API or analytics

> The browser crop/resize uses Canvas interpolation, which is not perfectly pixel-identical to Pillow/torchvision. Validate predictions against the Kaggle model on a representative set before reporting field results. Mobile performance also varies by device and browser.

## 1. Repository structure

```text
pharmavision-mobile-web/
├── README.md
├── .gitignore
├── vercel.json
├── tools/
│   └── export_trained_model_on_kaggle.ipynb
└── public/                        # Website deployed by Vercel
    ├── index.html
    ├── styles.css
    ├── app.js
    ├── favicon.svg
    └── model/
        ├── PLACE_ONNX_HERE.txt
        └── efficientnetv2_s.onnx  # YOU create and add this file
```

**The `.onnx` file is not in the starter ZIP.** This is deliberate: only your existing saved 81.63 MB `.pth` checkpoint has the actual learned weights. A web app with no model may load visually, but cannot provide predictions.

## 2. Export your trained Kaggle checkpoint to ONNX

Use your original trained checkpoint `best_efficientnetv2_s.pth` (checkpoint epoch 18), **not** new ImageNet weights.

1. Open [Kaggle Code](https://www.kaggle.com/code). Import `tools/export_trained_model_on_kaggle.ipynb` as a new notebook (the import option may be in **File**).
2. Click **Add Input** and attach your private **PharmaVision Trained Model Backup** dataset that contains `best_efficientnetv2_s.pth`. If it is already in your running Kaggle session under `/kaggle/working/pharmavision_reconstruction/`, the exporter can find it there too.
3. Run the cells **in order**. No GPU is required for conversion, and there is **no retraining**.
4. The notebook creates `/kaggle/working/efficientnetv2_s.onnx`. Download that file via the provided `FileLink`, or via Kaggle's Output panel.
5. On your computer, copy **`efficientnetv2_s.onnx`** into **`public/model/`**.

**Conversion checks:** The exporter tests ONNX Runtime CPU logits versus PyTorch logits for one dummy tensor. This checks graph conversion; it does **not** verify that a browser's Canvas crop matches `torchvision` preprocessing. For scientific use, check several *actual tablet images* in both environments.

## 3. Upload the entire project to GitHub

**Important file-size issue:** GitHub's web upload accepts individual files up to 25 MiB. The ONNX model is likely about 80 MB, so you **cannot** upload that one file from GitHub's web file uploader. Use **GitHub Desktop** or command-line Git to commit and push. GitHub warns for files over 50 MiB and blocks files over 100 MiB. Vercel's Hobby plan has a 100 MB static file upload limit. Check the actual ONNX file size first. If it exceeds 100 MiB, do **not** attempt this specific hosting route; use alternate model hosting / a smaller verified model.

Beginner GitHub Desktop workflow:

1. Make a new **empty repository** on [github.com/new](https://github.com/new) named `pharmavision-mobile-web`. Choose **Private** while developing. Do not add a README through GitHub (this ZIP already has one).
2. Install [GitHub Desktop](https://desktop.github.com/), sign in, and choose **File → Clone repository** to a folder on your computer.
3. Unzip this starter ZIP. Copy its **contents** (not the outer folder) into the cloned repository's folder.
4. Put your generated `efficientnetv2_s.onnx` inside the cloned folder's `public/model/`.
5. In GitHub Desktop, verify the file list, enter a summary such as `Initial tablet inspection prototype`, click **Commit to main**, then **Push origin**.
6. Refresh GitHub in your browser. Confirm that `public/index.html`, `public/app.js`, and `public/model/efficientnetv2_s.onnx` appear.

The private `.pth` checkpoint, Kaggle image dataset and personal experiments do **not** need to be placed in this public deployment repository. Obtain permission before publishing model weights if you are using someone else's trained checkpoint or restricted dataset.

## 4. Deploy with Vercel

1. Visit [vercel.com/new](https://vercel.com/new) and sign in using GitHub.
2. Import the **`pharmavision-mobile-web`** GitHub repository.
3. Set **Framework Preset:** `Other`. **Root Directory:** `./`. **Output Directory:** `public`. There is **no build command**, install command, Node runtime or environment variable required.
4. Click **Deploy**. Vercel provides an HTTPS URL for mobile and desktop.
5. Open the URL on your phone. Choose **Take photo**, frame a tablet and crop it. Press **Analyze image**; the first use downloads the model, so Wi-Fi is recommended.

If you see *Model file not found*, check that the `.onnx` file is named **exactly** `efficientnetv2_s.onnx` under `public/model/`, pushed to GitHub and included in the Vercel deployment. Test `https://YOUR-SITE.vercel.app/model/efficientnetv2_s.onnx` — it should return a model file, not a 404 page.

If Vercel reports a static file size error, check model size and your plan's limits. If your phone fails to load it or inference is too slow, use a backend-hosted model or a **validated** smaller/optimized ONNX export. Do not quietly change the architecture and assume identical accuracy.

## 5. Local test (optional)

From inside the repository folder, with Python installed:

```bash
python -m http.server --directory public 8000
```

Open `http://localhost:8000`. Do not double-click `index.html` as a local file, because model loading requires an HTTP(S) origin.

## 6. Field testing checklist

1. Obtain approval/permission before photographing production products or working on a regulated site.
2. For each known tablet, record visually verified **normal / defective / uncertain** condition independently of the prediction.
3. Use one tablet per photograph, a plain background, multiple cameras and lighting conditions; make the crop include all visibly important edges/surface features.
4. Record predictions, false negatives, false positives, camera/browser, date and any disagreements. Use **Export session CSV** for the predictions. The app keeps the log only in current-page memory.
5. Compare mobile-browser probabilities with the original Kaggle PyTorch model on representative images; Canvas resizing can alter results.
6. Re-evaluate on a large independent dataset with representative defects before suggesting operational use.

### Known limitations

- The system is a **binary visual classifier**, not crack localization, pill recognition, contaminant testing, dose-uniformity testing or chemical analysis.
- It has not been validated for acceptance/rejection of real manufacturing batches or patient use. On independent pillQC evaluation, your previously measured model missed **14/43** defective tablets.
- The probability shown is a model score, **not a validated confidence of pharmaceutical safety**.
- Browser inference runs on one WebAssembly CPU thread. An ~80 MB model can be slow or fail on memory-constrained phones; the first load needs network access.
- No continuous video scanner, automatic tablet detection, or batch processing is implemented. This prototype uses still photos and manual cropping.

## Attribution

Independent interface and ONNX deployment for a **reconstructed** EfficientNetV2-S tablet inspection model. Original PharmaVision work: Mahmood et al., *Pharmaceutics* (2026), DOI [10.3390/pharmaceutics18101287](https://doi.org/10.3390/pharmaceutics18101287). Original dataset and related research code: [tahirlee/pharmaceutical-tablet-defect-detection](https://github.com/tahirlee/pharmaceutical-tablet-defect-detection). Check the upstream dataset and code licenses before redistribution.
