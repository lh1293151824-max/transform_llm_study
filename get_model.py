import os

from huggingface_hub import snapshot_download


# Use the Hugging Face mirror.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

MODEL_ID = "Qwen/Qwen2.5-1.5B"
LOCAL_DIR = "./model"

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    resume_download=True,
)

print(f"Downloaded {MODEL_ID} to {LOCAL_DIR}")
