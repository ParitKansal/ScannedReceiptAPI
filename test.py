import requests

# URL
url = "http://localhost:8000/predict"

# List of local image paths (you can include one or many)
file_paths = [
    "/Users/xelpmoc/Downloads/random_crop_rot0_103_13_8.FRIENDS UNITED LTD - RECEIPTS (JUNE) 6_page_5_filled.jpg",
    "/Users/xelpmoc/Downloads/random_crop_rot0_103_13_8.FRIENDS UNITED LTD - RECEIPTS (JUNE) 6_page_5_filled.jpg",
]

# Prepare files for upload
files = [("files", open(path, "rb")) for path in file_paths]

# Send POST request
response = requests.post(url, files=files)

# Close files
for _, f in files:
    f.close()

# Print JSON response
print(response.json())