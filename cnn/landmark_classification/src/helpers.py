import os
import urllib.request
from io import BytesIO
from zipfile import ZipFile


def get_data_location():
    """
    Find the location of the dataset in `cnn/landmark_classification/data`.
    If not found, return the intended download location.
    """
    data_folder = "cnn/landmark_classification/data"
    if not os.path.exists("cnn/landmark_classification/data"):
        os.makedirs(data_folder, exist_ok=True)  # Ensure the path exists for download
    return data_folder


def download_and_extract(
    url="https://udacity-dlnfd.s3-us-west-1.amazonaws.com/datasets/landmark_images.zip",
):
    location = get_data_location()

    if not os.path.exists(os.path.join(location, "landmark_images`")):
        # Dataset does not exist; download and extract it
        print(f"Downloading and unzipping {url}. This will take a while...")

        with urllib.request.urlopen(url) as resp:
            with ZipFile(BytesIO(resp.read())) as fp:
                fp.extractall(location)

        print(f"Done. Dataset extracted to {location}")
    else:
        print(
            "Dataset already downloaded. If you need to re-download, "
            f"please delete the directory {location}"
        )


if __name__ == "__main__":
    download_and_extract()
