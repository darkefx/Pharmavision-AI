# PharmaVision AI

**AI-assisted visual screening of pharmaceutical tablets**

[**Live demo**](https://pharmavision-ai-zeta.vercel.app/) · [Research paper](https://doi.org/10.3390/pharmaceutics18101287) · [Original PharmaVision dataset and benchmark](https://github.com/tahirlee/pharmaceutical-tablet-defect-detection)

PharmaVision AI is a responsive research prototype that classifies photographs of individual pharmaceutical tablets as **visually normal** or **visually defective**. It uses an independently reconstructed and trained **EfficientNetV2-S** image classifier based on the published PharmaVision benchmark.

The website runs in desktop and mobile browsers. Users can choose or capture a photograph, manually crop the tablet, and obtain a prediction with an estimated defect probability. It also supports recording observations and exporting a session log as CSV.

> **Research demonstration only.** This application is not a validated pharmaceutical quality-control device. Its predictions must not be used alone to approve or reject medicines, release manufacturing batches, or make patient-safety decisions.

## How it works

1. Capture or upload an image containing **one tablet**.
2. Crop around the tablet, keeping its edges and visible surface features in frame.
3. The application resizes the crop to **224 × 224**, applies ImageNet normalization, and runs the trained model.
4. The model predicts **normal** or **defective** using a **0.40 threshold** on the defective-class probability.
5. Review the prediction and, if relevant, record the observed condition for later comparison.

The model detects *visual* abnormalities in photographs; it does not measure chemical composition, drug potency, dissolution, dose uniformity, or contamination not visible in the image. It does not identify or localize individual defect types.

## Model and evaluation

| Item | Details |
|---|---|
| Architecture | EfficientNetV2-S, initialized with ImageNet pretrained weights |
| Training framework | PyTorch / Torchvision |
| Deployment format | ONNX, via ONNX Runtime Web |
| Classes | `defective` (index 0), `normal` (index 1) |
| Input | Cropped RGB image, 224 × 224 |
| Model selection | Best validation F2-score; checkpoint at epoch 18 |
| Training | Up to 100 epochs; early stopped at epoch 33 |

The following are the results of **this project's reconstructed implementation**, not the original paper's released model:

| Dataset | Images evaluated | Accuracy | Defective-class precision | Defective-class recall | Defective-class F1 |
|---|---:|---:|---:|---:|---:|
| PharmaVision held-out test | 200 | **95.00%** | 94.12% | 96.00% | 95.05% |
| pillQC external test (chip and normal only) | 192 | **90.62%** | 87.88% | 67.44% | 76.32% |

On the **pillQC** external test, the model detected **29 of 43 chipped tablets** and missed **14**. That drop in defective-class recall is a significant practical limitation, even though overall accuracy remained high. These measurements were obtained in Kaggle using the saved PyTorch checkpoint. **The deployed browser implementation has not yet been independently verified to reproduce those metrics** across devices and photo-preprocessing conditions.

For context, the published study benchmarked 12 architectures and reported an **88.56% F1-score** for EfficientNetV2-S on PharmaVision and **75.00%** on pillQC. Differences should not be interpreted as proof of superiority: the training implementation and experimental conditions are not identical, and the test sets are limited in size.

## Website and technical implementation

- **Interface:** HTML, CSS and JavaScript; responsive for mobile and desktop.
- **Inference:** ONNX Runtime Web with WebAssembly execution **in the visitor's browser**; no separate prediction API in the current deployment.
- **Images:** Inference is performed locally in the browser. Photographs are not sent to an AI inference server by this version of the application.
- **Model size:** Approximately 78 MB, downloaded by the browser when required. Subsequent visits may use browser caching.
- **Field records:** Optional session logging and CSV export; do not assume a record is permanently stored unless exported.

**Known deployment limitation:** An approximately 78 MB model may initialize slowly on desktop hardware and may fail with WebAssembly memory errors on some mobile browsers. A server-hosted inference API is a potential future improvement; **it is not implemented in this version**.

## Responsible field evaluation

Use independent, human-verified reference labels and test across different tablets, lighting conditions, backgrounds, camera models, and defect severities. Record false negatives and false positives separately. Treat ambiguous predictions as cases for expert review, not automatic acceptance.

This is a research and demonstration tool intended to **support**, not replace, qualified pharmaceutical quality-assurance and quality-control personnel.

## Research attribution

This project is an **independent educational reconstruction and web demonstration** inspired by:

Mahmood, T., Rehman, S. U., Niamat, A., Lee, S. J., & Park, K. R. (2026). *Artificial Intelligence-Based Quality Control of Pharmaceutical Tablets: A Visual Dataset and Benchmark for Automated Defect Detection*. **Pharmaceutics, 18**, 1287. https://doi.org/10.3390/pharmaceutics18101287

- [Original benchmark repository and PharmaVision data](https://github.com/tahirlee/pharmaceutical-tablet-defect-detection)
- [PharmaVision dataset archive](https://zenodo.org/records/22773411)
- [pillQC external evaluation dataset](https://github.com/matlab-deep-learning/pillQC)

The reconstructed training pipeline, trained checkpoint, and website **should not be represented as the original authors' official deployment**. All evaluation figures above refer to this reconstruction unless explicitly identified as published results.
