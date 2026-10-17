# Deploying

Two free services:

| Part | Where | Why |
|---|---|---|
| Backend (FastAPI, 3 models, ~850 MB weights) | Hugging Face Spaces, Docker, free CPU (16 GB RAM) | Enough memory for VGG16 + U-Net + 3D SwinUNETR |
| Frontend (Next.js) | Vercel, free Hobby plan | Built for Next.js; deploys on every push to GitHub |

## 1. Backend → Hugging Face Space

```bash
hf auth login                                   # token with "write" access: huggingface.co/settings/tokens
deploy/deploy_backend_hf.sh <your-hf-username>/neuroscan-api
```

The Space builds for 5–10 minutes. Check `https://<your-hf-username>-neuroscan-api.hf.space/api/health`.
In the Space **Settings → Variables**, add `CORS_ORIGINS` = your Vercel URL from step 2.

## 2. Frontend → Vercel (from GitHub)

1. vercel.com/new → **Import** this GitHub repository.
2. **Root Directory**: `frontend`.
3. **Environment variable**: `NEXT_PUBLIC_API_URL` = the Space URL from step 1.
4. **Deploy**. Every push to `main` redeploys automatically.

## Notes

- The model weights are not in Git (GitHub's 100 MB file limit); they go straight to the Space in step 1.
- A free Space sleeps when idle; the first request then takes about a minute.
- A full-resolution 3D SwinUNETR run takes a few minutes on the 2-vCPU free Space; fast mode and the result cache help.
