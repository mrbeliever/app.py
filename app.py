import re
import streamlit as st
from PIL import Image
import numpy as np
import torch
from diffusers import FluxImg2ImgPipeline

# Set up model configuration
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe = FluxImg2ImgPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to(device)

def sanitize_prompt(prompt):
    allowed_chars = re.compile(r"[^a-zA-Z0-9\s.,!?-]")
    sanitized_prompt = allowed_chars.sub("", prompt)
    return sanitized_prompt

def convert_to_fit_size(original_width_and_height, maximum_size=2048):
    width, height = original_width_and_height
    if width <= maximum_size and height <= maximum_size:
        return width, height
    scaling_factor = maximum_size / max(width, height)
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return new_width, new_height

def adjust_to_multiple_of_32(width: int, height: int):
    width = width - (width % 32)
    height = height - (height % 32)
    return width, height

def process_images(image, prompt="a girl", strength=0.75, seed=0, inference_step=4):
    def process_img2img(image, prompt="a person", strength=0.75, seed=0, num_inference_steps=4):
        if image is None:
            st.warning("No input image provided.")
            return None
        generator = torch.Generator(device).manual_seed(seed)
        width, height = convert_to_fit_size(image.size)
        width, height = adjust_to_multiple_of_32(width, height)
        image = image.resize((width, height), Image.LANCZOS)
        output = pipe(prompt=prompt, image=image, generator=generator, strength=strength, width=width, height=height,
                      guidance_scale=0, num_inference_steps=num_inference_steps, max_sequence_length=256)
        return output.images[0]
    return process_img2img(image, prompt, strength, seed, inference_step)

# Streamlit App Layout
st.title("Image-to-Image Generation")

# File uploader for the input image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_image:
    input_image = Image.open(uploaded_image)
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

prompt = st.text_input("Prompt", value="a woman", help="Your prompt for image generation.")
strength = st.slider("Strength", min_value=0.0, max_value=0.75, value=0.75, step=0.01)
seed = st.number_input("Seed", value=100, min_value=0, step=1)
inference_step = st.number_input("Inference Steps", value=4, min_value=1, step=1)

# Button to generate image
if st.button("Generate"):
    if uploaded_image:
        output_image = process_images(input_image, prompt, strength, seed, inference_step)
        if output_image:
            st.image(output_image, caption="Generated Image", use_column_width=True)
    else:
        st.warning("Please upload an image before generating.")

st.markdown("<hr>", unsafe_allow_html=True)
st.write("Thank you for using this Image-to-Image Generation app!")
