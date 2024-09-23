FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"

RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home && \
    apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    numpy==1.25.2 diffusers transformers rembg einops accelerate timm git+https://github.com/tencent-ailab/IP-Adapter && \
    GIT_LFS_SKIP_SMUDGE=1 git clone -b dev https://github.com/camenduru/zest_code /content/zest_code && \
    git clone https://huggingface.co/h94/IP-Adapter /content/IP-Adapter && mv /content/IP-Adapter/models /content/zest_code/models && mv /content/IP-Adapter/sdxl_models /content/zest_code/sdxl_models && \
    git clone https://github.com/tencent-ailab/IP-Adapter /content/zest_code/ip_adapter && git clone https://github.com/isl-org/DPT /content/zest_code/DPT && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt -d /content/zest_code/DPT/weights -o dpt_hybrid-midas-501f0c75.pt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/raw/main/config.json -d /content/diffusers/controlnet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.fp16.safetensors -d /content/diffusers/controlnet -o diffusion_pytorch_model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/model_index.json -d /content/sdxl -o model_index.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/scheduler/scheduler_config.json -d /content/sdxl/scheduler -o scheduler_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/text_encoder/config.json -d /content/sdxl/text_encoder -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder/model.fp16.safetensors -d /content/sdxl/text_encoder -o model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/text_encoder_2/config.json -d /content/sdxl/text_encoder_2 -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/text_encoder_2/model.fp16.safetensors -d /content/sdxl/text_encoder_2 -o model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/merges.txt -d /content/sdxl/tokenizer -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/special_tokens_map.json -d /content/sdxl/tokenizer -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/tokenizer_config.json -d /content/sdxl/tokenizer -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer/vocab.json -d /content/sdxl/tokenizer -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer_2/merges.txt -d /content/sdxl/tokenizer_2 -o merges.txt && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer_2/special_tokens_map.json -d /content/sdxl/tokenizer_2 -o special_tokens_map.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer_2/tokenizer_config.json -d /content/sdxl/tokenizer_2 -o tokenizer_config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/tokenizer_2/vocab.json -d /content/sdxl/tokenizer_2 -o vocab.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/unet/config.json -d /content/sdxl/unet -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/unet/diffusion_pytorch_model.fp16.safetensors -d /content/sdxl/unet -o diffusion_pytorch_model.fp16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/raw/main/vae_1_0/config.json -d /content/sdxl/vae -o config.json && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/vae_1_0/diffusion_pytorch_model.fp16.safetensors -d /content/sdxl/vae -o diffusion_pytorch_model.fp16.safetensors

COPY ./worker_runpod.py /content/zest_code/worker_runpod.py
WORKDIR /content/zest_code
CMD python worker_runpod.py