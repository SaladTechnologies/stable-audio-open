FROM docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip

RUN pip install gradio

RUN apt-get install -y net-tools

RUN pip install stable_audio_tools

RUN apt-get install -y libsndfile1

WORKDIR /app
COPY hello-gradio.py /app

CMD ["python", "hello-gradio.py"]
