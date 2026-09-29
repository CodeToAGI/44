# ep44_app.py
# Challenge: Deploy the full 3-tab Gradio app to Hugging Face Spaces
# Bonus: add a 4th style-transfer tab

import gradio as gr
import torch
from PIL import Image
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
)

# ── Load pipeline ONCE (shared across all tabs) ──────────────────────────────
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    safety_checker=None,          # optional — disable NSFW filter for demos
)
pipe = pipe.to("cuda")            # remove .to("cuda") for CPU

# Share weights — no extra download / VRAM
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pipe(pipe)
inpaint_pipe = StableDiffusionInpaintPipeline.from_pipe(pipe)


# ── Generate functions ───────────────────────────────────────────────────────
def generate_txt2img(prompt, neg, cfg, steps, seed):
    g = (torch.Generator(device="cuda").manual_seed(int(seed))
         if seed != -1 else None)
    return pipe(
        prompt=prompt,
        negative_prompt=neg,
        guidance_scale=cfg,
        num_inference_steps=steps,
        generator=g,
    ).images[0]


def generate_img2img(image_np, prompt, neg, strength, cfg, steps):
    if image_np is None:
        raise gr.Error("Please upload an image first.")
    img = Image.fromarray(image_np).convert("RGB")
    return img2img_pipe(
        prompt=prompt,
        image=img,
        strength=strength,
        negative_prompt=neg,
        guidance_scale=cfg,
        num_inference_steps=steps,
    ).images[0]


def generate_inpaint(editor_dict, prompt, neg, cfg, steps):
    if editor_dict is None:
        raise gr.Error("Please upload an image and draw a mask.")
    img  = Image.fromarray(editor_dict["image"]).convert("RGB")
    mask = Image.fromarray(editor_dict["mask"]).convert("RGB")
    return inpaint_pipe(
        prompt=prompt,
        image=img,
        mask_image=mask,
        negative_prompt=neg,
        guidance_scale=cfg,
        num_inference_steps=steps,
    ).images[0]


def generate_style_transfer(image_np, style_prompt, neg, strength, cfg, steps):
    """Bonus 4th tab — style transfer via img2img."""
    if image_np is None:
        raise gr.Error("Please upload an image first.")
    img = Image.fromarray(image_np).convert("RGB")
    full_prompt = f"{style_prompt}, highly detailed, masterpiece"
    return img2img_pipe(
        prompt=full_prompt,
        image=img,
        strength=strength,
        negative_prompt=neg,
        guidance_scale=cfg,
        num_inference_steps=steps,
    ).images[0]


# ── Gradio UI ────────────────────────────────────────────────────────────────
DEFAULT_NEG = "blurry, low quality, ugly, distorted, deformed, bad anatomy, watermark, signature, text"

with gr.Blocks(title="Text-to-Image Generator") as demo:
    gr.Markdown("# 🎨 Text-to-Image Generator\nStable Diffusion + Gradio · CodeToAGI EP44")

    with gr.Tabs():
        # ── Tab 1: Text to Image ─────────────────────────────────────────────
        with gr.TabItem("Text to Image"):
            with gr.Row():
                prompt_t = gr.Textbox(label="Prompt", lines=2,
                    placeholder="a golden retriever puppy in a sunlit meadow, oil painting...")
            with gr.Row():
                neg_t = gr.Textbox(label="Negative Prompt", lines=1, value=DEFAULT_NEG)
            with gr.Row():
                cfg_t   = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                steps_t = gr.Slider(10, 100, value=50, step=1, label="Steps")
                seed_t  = gr.Number(value=-1, label="Seed (-1 = random)")
            btn_t = gr.Button("✨ Generate Image", variant="primary")
            out_t = gr.Image(label="Generated Image")
            btn_t.click(generate_txt2img,
                        inputs=[prompt_t, neg_t, cfg_t, steps_t, seed_t],
                        outputs=out_t)

        # ── Tab 2: Image to Image ────────────────────────────────────────────
        with gr.TabItem("Image to Image"):
            img_i = gr.Image(label="Upload Image", type="numpy")
            prompt_i = gr.Textbox(label="Prompt", lines=2)
            neg_i = gr.Textbox(label="Negative Prompt", lines=1, value=DEFAULT_NEG)
            with gr.Row():
                strength_i = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Strength")
                cfg_i      = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                steps_i    = gr.Slider(10, 100, value=50, step=1, label="Steps")
            btn_i = gr.Button("✨ Transform Image", variant="primary")
            out_i = gr.Image(label="Transformed Image")
            btn_i.click(generate_img2img,
                        inputs=[img_i, prompt_i, neg_i, strength_i, cfg_i, steps_i],
                        outputs=out_i)

        # ── Tab 3: Inpainting ────────────────────────────────────────────────
        with gr.TabItem("Inpainting"):
            editor = gr.ImageEditor(label="Upload + Draw Mask", type="numpy")
            prompt_p = gr.Textbox(label="Prompt (what to fill)", lines=2)
            neg_p = gr.Textbox(label="Negative Prompt", lines=1, value=DEFAULT_NEG)
            with gr.Row():
                cfg_p   = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                steps_p = gr.Slider(10, 100, value=50, step=1, label="Steps")
            btn_p = gr.Button("✨ Inpaint", variant="primary")
            out_p = gr.Image(label="Inpainted Image")
            btn_p.click(generate_inpaint,
                        inputs=[editor, prompt_p, neg_p, cfg_p, steps_p],
                        outputs=out_p)

        # ── Tab 4: Style Transfer (Bonus) ────────────────────────────────────
        with gr.TabItem("Style Transfer"):
            img_s = gr.Image(label="Upload Photo", type="numpy")
            style_s = gr.Textbox(label="Style Prompt",
                value="in the style of Van Gogh, impressionist, vivid colours")
            neg_s = gr.Textbox(label="Negative Prompt", lines=1, value=DEFAULT_NEG)
            with gr.Row():
                strength_s = gr.Slider(0.0, 1.0, value=0.65, step=0.05, label="Strength")
                cfg_s      = gr.Slider(1, 20, value=7.5, step=0.5, label="CFG Scale")
                steps_s    = gr.Slider(10, 100, value=50, step=1, label="Steps")
            btn_s = gr.Button("✨ Apply Style", variant="primary")
            out_s = gr.Image(label="Styled Image")
            btn_s.click(generate_style_transfer,
                        inputs=[img_s, style_s, neg_s, strength_s, cfg_s, steps_s],
                        outputs=out_s)

demo.queue().launch()
