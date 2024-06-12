# Demo: Stable-Audio-Open-1.0 using Gradio

https://huggingface.co/stabilityai/stable-audio-open-1.0
https://www.gradio.app/

The Gradio code is configured to listen on an IPv6 port for running on SaladCloud; use IPv4 for local runs and tests:

demo.launch(server_name="[::]", server_port=8000), for IPv6 usage
demo.launch(server_name="0.0.0.0", server_port=8000), for IPv4 usage

# Deploy on SaladCloud

Image Source: saladtechnologies/richardx:0.0.3-stable (our prebuilt image or yours)
Replica Count: 1 (can only be one for this UI app)
Resource：2 vCPU, 12 GB Memory, any GPU types with 16 GB or more VRAM
Container Gateway: Enbaled, Port 8000
Environment Variables: HF_TOKEN (YOUR_HUGGINGFACE_ACCESS_TOKEN_READ)

The model weights (around 6 GB) will be dynamically downloaded from Hugging Face when the container is running.

# Local Run

### Build & Push
docker image build -t YOUR_IMAGE_NAME -f Dockerfile .
docker login
docker push YOUR_IMAGE_NAME 

### Run
docker run -it --rm -p 8000:8000 --gpus all \
 -e HF_TOKEN="YOUR_HUGGINGFACE_ACCESS_TOKEN_READ" \
 YOUR_IMAGE_NAME

### Run with the pre-downloaded model weights on the host
docker run -it --rm -p 8000:8000 --gpus all \
 -v /home/CURRENT_USER/.cache/huggingface:/root/.cache/huggingface \
 -e HF_TOKEN="HUGGINGFACE_ACCESS_TOKEN_READ" \
 YOUR_IMAGE_NAME


