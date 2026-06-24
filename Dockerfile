FROM python:3.10-slim

# --- System dependencies required by Manim + ffmpeg for video rendering ---
# (pango/cairo are needed to build manimpango & pycairo; ffmpeg is needed
# to actually encode the rendered frames into an .mp4)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libpango1.0-dev \
    libcairo2-dev \
    pkg-config \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
WORKDIR $HOME/app
COPY --chown=user requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt
COPY --chown=user . $HOME/app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
