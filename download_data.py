import os
import urllib.request

# Create the folder structure if it doesn't exist
os.makedirs("almrrc2021/almrrc2021-data-training/model_build_inputs", exist_ok=True)

base_url = "https://amazon-last-mile-challenges.s3.amazonaws.com/almrrc2021/almrrc2021-data-training/model_build_inputs/"
files = ["route_data.json", "package_data.json", "actual_sequences.json"]

for f in files:
    url = base_url + f
    dest = os.path.join(
        "almrrc2021", "almrrc2021-data-training", "model_build_inputs", f
    )
    print(f"Downloading {f}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Successfully downloaded {f}")

print("\nAll files downloaded successfully!")
