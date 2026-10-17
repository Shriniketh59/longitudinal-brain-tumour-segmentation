---
title: NeuroScan AI API
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-4.0
short_description: Brain tumor MRI classification, segmentation, 3D and longitudinal API
---

# NeuroScan AI · backend API

FastAPI service behind the NeuroScan AI web app. Research and education only, not for clinical use.

- `POST /api/classify`: VGG16 tumor type, cancer-risk estimate, Grad-CAM
- `POST /api/segment`: 2D U-Net (VGG16 encoder)
- `POST /api/video/analyze`: detection, segmentation and classification of video frames
- `GET /api/volume/samples/{id}?model=unet2d|swin3d`: 3D volumes, incl. 3D SwinUNETR
- `POST /api/longitudinal/analyze`: growth rate, response category, forecast
- `POST /api/report`: PDF report
- `GET /api/health`, `GET /api/models/metrics`

Set `CORS_ORIGINS` in the Space settings (Variables) to your frontend URL, e.g. `https://your-app.vercel.app`.

Data: BraTS 2023 Adult Glioma (CC BY 4.0) and the Nickparvar Brain Tumor MRI dataset (CC BY 4.0).
