import requests

# URL without save_input
url = "http://localhost:8000/predict"

# Local path to the image
file_path = "/Users/xelpmoc/Downloads/random_crop_rot0_103_13_8.FRIENDS UNITED LTD - RECEIPTS (JUNE) 6_page_5_filled.jpg"

# Open the file and send POST request
with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

# Print the JSON response
print(response.json())
