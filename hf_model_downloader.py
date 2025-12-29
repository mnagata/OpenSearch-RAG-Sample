import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_NAME = os.environ["MODEL_NAME"]

download_path = snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=f"{MODEL_DIR}/{MODEL_NAME}",
    local_dir_use_symlinks=False,  # ※1
)
