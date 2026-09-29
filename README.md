# EP44 — Build a Text-to-Image App with Stable Diffusion & Gradio

**Deep Learning Series · Episode 44 · Module 8 — Generative Deep Learning (PROJECT)**

Take everything from EP43 and turn it into a complete, shareable web application deployed for free on Hugging Face Spaces.

## What you'll build

- **Text-to-Image tab** — prompt, negative prompt, CFG slider, steps, seed
- **img2img tab** — upload a photo + strength slider + transform with a prompt
- **Inpainting tab** — draw a mask with `gr.ImageEditor` and fill the region
- **Bonus Style Transfer tab** — apply artistic styles (Van Gogh, etc.) to any photo
- **Deploy live** to Hugging Face Spaces (or `share=True` for a temporary public link)

## Quick start (local)

```bash
pip install diffusers transformers gradio accelerate torch Pillow
python ep44_app.py
