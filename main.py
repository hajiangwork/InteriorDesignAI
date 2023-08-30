from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import cv2
import random
from datetime import datetime
from PIL import Image
import numpy as np
import gradio as gr

from annotator.util import resize_image, HWC3
from annotator.mlsd import MLSDdetector
from chatgpt import prompt_gen

# pretrained_controlnet = 'lllyasviel/sd-controlnet-mlsd'
pretrained_controlnet = 'lllyasviel/control_v11p_sd15_mlsd'
controlnet = ControlNetModel.from_pretrained(pretrained_controlnet, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
apply_mlsd = MLSDdetector()


def process(input_image, prompt, a_prompt, n_prompt, num_samples, strength, ddim_steps, image_resolution, return_lines, 
            scale=10, seed=None, value_threshold=0.1, distance_threshold=0.1):
    # log current time and prompt
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Prompt: {prompt}")
    
    strength *= 2  # raw model strength from 0 - 2, here we input 0 - 1
    with torch.no_grad():
        input_image = HWC3(input_image)
        img = resize_image(input_image, image_resolution)
        detected_map = apply_mlsd(img, value_threshold, distance_threshold)
        detected_map = HWC3(detected_map)
        model_input = Image.fromarray(detected_map)

        if seed is None:
            seed = random.randint(0, 2147483647)
        generator = torch.manual_seed(seed)

    out_images = pipe(
        prompt=prompt + ', ' + a_prompt, 
        num_images_per_prompt=num_samples,
        controlnet_conditioning_scale=strength,
        num_inference_steps=ddim_steps, 
        guidance_scale=scale,
        negative_prompt=n_prompt,
        generator=generator, 
        image=model_input
    ).images

    results = [255 - cv2.dilate(detected_map, np.ones(shape=(3, 3), dtype=np.uint8), iterations=1)] if return_lines else []
    # results += [cv2.cvtColor(np.asarray(out_image), cv2.COLOR_BGR2RGB) for out_image in out_images]
    results += [np.asarray(out_image) for out_image in out_images]
    return results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Interior design with Stable Diffusion and ControlNet")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', type="numpy")

            room_type = gr.Dropdown(
                label="Room Type", 
                choices=['Living room', 'Bedroom', 'Bathroom', 'Dining room', 'Kitchen', 'Study room'], 
                value='Living room')
                
            style = gr.Dropdown(
                label="Style", 
                choices=['Contemporary', 'Modern', 'Minimalist', 'Light Luxury', 'Rustic', 'Vintage', 'Industrial', \
                            'Resplendent', 'Cozy', 'City', 'Biophilic', 'Medieval', 'Outdoor Patio', \
                            'Scandinavian', 'Chinese', 'Japanese', 'Bohemian', 'Tropical', 'Maximalist', 'Gen Z'], 
                value='Contemporary')
            
            prompt_button = gr.Button(label="generate prompt", value="Generate a prompt")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run", value="Design my room")

            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Image Number", minimum=1, maximum=4, value=1, step=1)
                strength = gr.Slider(label="Reference Strength", minimum=0.0, maximum=1.0, value=0.4, step=0.01)
                a_prompt = gr.Textbox(label="Added Prompt", value='professional interior design, elegant, highly detailed, professional photography')
                n_prompt = gr.Textbox(label="Negative Prompt", value='dirty, ugly, sand, soil, clay, text, showy, ostentatious, exaggerated, anime style, low quality')
                ddim_steps = gr.Slider(label="Rendering Steps", minimum=1, maximum=100, value=20, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=768, value=512, step=64)
                return_lines = gr.Checkbox(label="Show sturcture lines", value=False)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')

    prompt_button.click(prompt_gen, inputs=[room_type, style], outputs=[prompt])

    ips = [input_image, prompt, a_prompt, n_prompt, num_samples, strength, ddim_steps, image_resolution, return_lines]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')
