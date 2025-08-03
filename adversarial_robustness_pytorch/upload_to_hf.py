import os
import shutil
from huggingface_hub import HfApi, HfFolder, Repository

# Authenticate with Hugging Face
HF_TOKEN = os.getenv("HF_TOKE")  # Use environment variable for the token
if not HF_TOKEN:
    raise ValueError("Hugging Face token not found. Please set the HF_TOKEN environment variable.")
HfFolder.save_token(HF_TOKEN)
api = HfApi()

# Define the repository name and paths for the weights
model_repo_name = "Hayleyy2024/adv"  # Replace with your username or organization name
model_weights = {
    "5-class-vehicle_parx": "/workspace/Treenet/adversarial_robustness_pytorch/par_x/5-class-vehicle_parx/weights-best.pt",
    "7-class-animal_parx": "/workspace/Treenet/adversarial_robustness_pytorch/par_x/7-class-animal_parx/weights-best.pt",
}

# Function to upload all weights into a single repository
def upload_all_weights(repo_name, weights):
    print(f"Accessing repository: {repo_name}")
    repo_url = f"https://huggingface.co/{repo_name}"
    repo = Repository(local_dir="combined_repo", clone_from=repo_url)
    
    try:
        # Copy each weight file into a subdirectory within the repository
        for model_name, weight_path in weights.items():
            if not os.path.exists(weight_path):
                print(f"Weight file for {model_name} not found: {weight_path}")
                continue
            
            subdir = os.path.join("combined_repo", model_name)
            os.makedirs(subdir, exist_ok=True)
            dest_path = os.path.join(subdir, os.path.basename(weight_path))
            shutil.copy(weight_path, dest_path)
            print(f"Copied {weight_path} to {dest_path}")
        
        # Push all changes to Hugging Face Hub
        repo.push_to_hub(commit_message="Upload all model weights")
        print(f"Successfully uploaded all weights to {repo_name}.")
    except Exception as e:
        print(f"An error occurred while uploading weights: {e}")
    finally:
        # Clean up local repository folder
        if os.path.exists("combined_repo"):
            shutil.rmtree("combined_repo")
            print("Cleaned up local repository folder: combined_repo")

# Upload all weights to a single repository
upload_all_weights(model_repo_name, model_weights)
