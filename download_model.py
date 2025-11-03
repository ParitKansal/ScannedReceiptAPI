import gdown

# Google Drive file ID from your link
file_id = "1Qd9xthmo5cUYrl5U34l9lB9MtUgr-mhl"
output_path = "model.pt"  # local path to save the model

# Download
gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)
print("✅ Model downloaded to", output_path)
