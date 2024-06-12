import torch
import torchaudio
from einops import rearrange
import gradio as gr
import os
import uuid
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond


device = "cuda" if torch.cuda.is_available() else "cpu"

# Download model
model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
sample_rate = model_config["sample_rate"]
sample_size = model_config["sample_size"]

model = model.to(device)



def generate_audio(prompt, seconds_total=30, steps=100, cfg_scale=7):

    # Set up text and timing conditioning
    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": seconds_total
    }]
    print(f"Conditioning: {conditioning}")

    # Generate stereo audio
    print("Generating audio...")
    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=0.3,
        sigma_max=500,
        sampler_type="dpmpp-3m-sde",
        device=device
    )
    print("Audio generated.")

    # Rearrange audio batch to a single sequence
    output = rearrange(output, "b d n -> d (b n)")
    print("Audio rearranged.")

    # Peak normalize, clip, convert to int16
    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    print("Audio normalized and converted.")

    # Generate a unique filename for the output
    unique_filename = f"output_{uuid.uuid4().hex}.wav"
    print(f"Saving audio to file: {unique_filename}")

    # Save to file
    torchaudio.save(unique_filename, output, sample_rate)
    print(f"Audio saved: {unique_filename}")

    # Return the path to the generated audio file
    return unique_filename


# Setting up the Gradio Interface
demo = gr.Interface(
    fn=generate_audio,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your text prompt here"),
        gr.Slider(0, 47, value=30, label="Duration in Seconds"),
        gr.Slider(10, 150, value=100, step=10, label="Number of Diffusion Steps"),
        gr.Slider(1, 15, value=7, step=0.1, label="CFG Scale")
    ],
    outputs=gr.Audio(type="filepath", label="Generated Audio"),
    title="Stable Audio Generator on SaladCloud",
    description="Generate variable-length stereo audio at 44.1kHz from text prompts using Stable Audio Open 1.0.",
    examples=[
    [
        "Create a serene soundscape of a quiet beach at sunset.",  # Text prompt

        45,  # Duration in Seconds
        100,  # Number of Diffusion Steps
        10,  # CFG Scale
    ],
    [
        "Generate an energetic and bustling city street scene with distant traffic and close conversations.",  # Text prompt

        30,  # Duration in Seconds
        120,  # Number of Diffusion Steps
        5,  # CFG Scale
    ],
    [
        "Simulate a forest ambiance with birds chirping and wind rustling through the leaves.",  # Text prompt
        60,  # Duration in Seconds
        140,  # Number of Diffusion Steps
        7.5,  # CFG Scale
    ],
    [
        "Recreate a gentle rainfall with distant thunder.",  # Text prompt

        35,  # Duration in Seconds
        110,  # Number of Diffusion Steps
        8,  # CFG Scale

    ],
    [
        "Imagine a jazz cafe environment with soft music and ambient chatter.",  # Text prompt
        25,  # Duration in Seconds
        90,  # Number of Diffusion Steps
        6,  # CFG Scale

    ],
    ["Rock beat played in a treated studio, session drumming on an acoustic kit.",
        30,  # Duration in Seconds
        100,  # Number of Diffusion Steps
        7,  # CFG Scale

    ]
])

demo.launch(server_name="[::]", server_port=8000)
#demo.launch(server_name="0.0.0.0", server_port=8000)