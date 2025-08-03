import os
import shutil
from huggingface_hub import hf_hub_download

# Authenticate with Hugging Face
HF_TOKEN = os.getenv("HF_TOKE")  # Use environment variable for the token
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")

# Define the repository name and paths for the weights
model_repo_name = "Hayleyy2024/adv"  # Replace with your username or organization name
model_weights = {
    "5-class-vehicle_parx": "5-class-vehicle_parx/weights-best.pt",
    "7-class-animal_parx": "7-class-animal_parx/weights-best.pt",
}

# Function to download all weights from the repository
def download_all_weights(repo_name, weights, output_dir="downloaded_weights"):
    os.makedirs(output_dir, exist_ok=True)
    try:
        for model_name, weight_path in weights.items():
            print(f"Downloading {model_name} from repository: {repo_name}")
            local_path = hf_hub_download(
                repo_id=repo_name,
                filename=weight_path,
                token=HF_TOKEN,
                cache_dir=output_dir
            )
            dest_path = os.path.join(output_dir, model_name, os.path.basename(weight_path))
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            shutil.move(local_path, dest_path)
            print(f"Downloaded {model_name} to {dest_path}")
    except Exception as e:
        print(f"An error occurred while downloading weights: {e}")

# Download all weights from the repository
download_all_weights(model_repo_name, model_weights)
