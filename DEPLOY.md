# Deploying

Two free services:

| Part | Where | Why |
|---|---|---|
| Backend (FastAPI, 3 models, ~850 MB weights) | **Modal** (serverless, free monthly credits) | 6 GB containers fit VGG16 + U-Net + 3D SwinUNETR; scales to zero when idle |
| Frontend (Next.js) | Vercel, free Hobby plan | Built for Next.js; deploys on every push to GitHub |

## 1. Backend → Modal

```bash
pip install modal && python -m modal setup            # browser login, once
cd backend
python -m modal volume create neuroscan-weights
python -m modal volume put neuroscan-weights weights/vgg16_classifier.pt /vgg16_classifier.pt   # repeat for each .pt / .metrics.json
python -m modal run modal_app.py::download_samples    # BraTS sample cases for the 3D lab
python -m modal deploy modal_app.py                   # prints the https://...modal.run URL
```

Check `<url>/api/health`. Allowed frontend origins are set in `FRONTEND_ORIGINS` in `backend/modal_app.py`.

(Alternative: `deploy/deploy_backend_hf.sh` for a Hugging Face Docker Space, which now requires HF PRO.)

## 2. Frontend → Vercel (from GitHub)

1. vercel.com/new → **Import** this GitHub repository.
2. **Root Directory**: `frontend`.
3. **Environment variable**: `NEXT_PUBLIC_API_URL` = the Modal URL from step 1.
4. **Deploy**. Every push to `main` redeploys automatically.

## Notes

- The model weights are not in Git (GitHub's 100 MB file limit); they go straight to the Modal volume in step 1.
- Modal scales to zero when idle; the first request after a pause takes ~15–20 s (cold start).
- On Modal's 4 vCPUs a 3D SwinUNETR run takes ~20 s in fast mode.
