# React frontend (Vercel-ready)

This replaces the Streamlit UI in `../frontend/` with a plain Vite + React app.
It talks to the exact same FastAPI backend over HTTP and Server-Sent Events —
no backend changes are required.

## Local dev

```bash
cd frontend-react
npm install
cp .env.example .env   # then set VITE_BACKEND_URL to your local or Render backend
npm run dev
```

## Deploy to Vercel

1. Import this GitHub repo into Vercel (or connect it if already imported).
2. Set **Root Directory** to `frontend-react`. Vercel auto-detects the Vite
   framework preset (build command `npm run build`, output directory `dist`).
3. Add an environment variable:
   - `VITE_BACKEND_URL` = your Render backend URL, e.g.
     `https://resume-analyzer-app-w0hb.onrender.com`
   - optionally `VITE_GITHUB_REPOSITORY_URL` if you want the "cloud demo"
     banner to link somewhere other than this repo.
4. Deploy.

The Render backend's `CORS_ORIGINS` env var should include your Vercel domain
(or stay at the default `*` for a quick demo).

## What's different from the Streamlit version

- No in-process fallback: if the backend is unreachable, this UI shows an
  error instead of running the LangGraph pipeline in the browser (Streamlit's
  Python runtime could do that; a static React build cannot).
- Same endpoints, same data shapes, same score/gaps/improvements/preparation/
  keywords model as the FastAPI backend already returns.
