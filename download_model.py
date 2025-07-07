import os
import requests

HF_URL = "https://huggingface.co/TechGeeek/TripoSR-model/resolve/main/TripoSR.ckpt"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "tripoSR.ckpt")

def download():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    if os.path.exists(CHECKPOINT_PATH):
        print("Checkpoint already exists.")
        return
    print("Downloading model from Hugging Face...")
    r = requests.get(HF_URL, stream=True)
    r.raise_for_status()
    with open(CHECKPOINT_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

if __name__ == "__main__":
    download()
