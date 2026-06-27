# Deployment — Hugging Face Spaces (Docker)

The app ships as one container (FastAPI + the static frontend + the committed
`models/baseline.joblib`). Hugging Face Spaces runs the `Dockerfile` directly and
reads the Space config (`sdk: docker`, `app_port: 7860`) from YAML frontmatter at
the top of `README.md`.

GitHub renders that frontmatter as an ugly table, so `README.md` on `main` has
none — the config lives in `deploy/hf-header.md` and `scripts/deploy-hf.sh`
prepends it only for the Space push (see **Deploy** below).

## Local check (optional, needs Docker running)

```bash
docker build -t soccerboard .
docker run --rm -p 7860:7860 soccerboard
# open http://localhost:7860
```

## Deploy

1. Create a Space at https://huggingface.co/new-space
   - **SDK: Docker** (blank template)
   - note the repo URL: `https://huggingface.co/spaces/<user>/SoccerBoard`

2. Push this repo to the Space:

   ```bash
   pip install huggingface_hub
   huggingface-cli login                      # paste a write token from hf.co/settings/tokens
   git remote add space https://huggingface.co/spaces/<user>/SoccerBoard
   ./scripts/deploy-hf.sh                      # prepends HF config header, force-pushes to the Space
   ```

   The script requires a clean working tree; it force-pushes the
   frontmatter-prefixed README to the Space and leaves your local `main`
   untouched.

3. The Space builds the image and goes live at
   `https://huggingface.co/spaces/<user>/SoccerBoard`. First build takes a few
   minutes; subsequent pushes rebuild automatically.

## Notes

- The model artifact is committed (`models/baseline.joblib`), so the build is
  offline and fast — it does **not** pull StatsBomb data or retrain.
- To ship a retrained model: `python -m xg.models.baseline`, commit the updated
  `models/baseline.joblib`, and push.
- Runtime deps are the slim `requirements-serve.txt` (no torch / pandas / EDA
  tooling), which keeps the image small.
