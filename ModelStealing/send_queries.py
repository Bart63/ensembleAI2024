import requests
import json
import os
import numpy as np
def model_stealing(path_to_png_file: str):
    SERVER_URL = "http://34.71.138.79:9090"
    ENDPOINT = "/modelstealing"
    URL = SERVER_URL + ENDPOINT

    TEAM_TOKEN = "5KogzTO5QjSdXupe"

    with open(path_to_png_file, "rb") as img_file:
        response = requests.get(
            URL, files={"file": img_file}, headers={"token": TEAM_TOKEN}
        )

        if response.status_code == 200:
            return json.loads(response.content.decode())["representation"]
        else:
            raise Exception(f"Request failed. Status code: {response.status_code}, content: {response.content}")


def model_stealing_with_checkpoint(directory, n_queries=5, checkpoint_interval=100, save_path='feature_representations_sorted.npy'):
    """
    Send multiple queries to the API for each saved image in ordered fashion, with checkpointing.

    :param directory: Directory containing the saved images.
    :param n_queries: Number of queries to send for each image.
    :param checkpoint_interval: Number of images after which to save a checkpoint.
    :param save_path: Path to save the feature representations array.
    :return: Ordered list of feature representations corresponding to the original image array.
    """
    # Attempt to load existing progress if available
    files = [f for f in os.listdir(directory) if f.endswith(".png")]
    files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    if os.path.exists(save_path):
        feature_representations = np.load(save_path)
        start_index = np.max(np.where(~np.all(feature_representations == 0, axis=(1,2)))) + 1
        print(f"Resuming from image index {start_index}")
    else:
        feature_representations = np.zeros((len(files), n_queries, 512))  # Assuming 512 features per representation
        start_index = 0

    for i in range(start_index, len(files)):
        filename = files[i]
        path_to_png_file = os.path.join(directory, filename)
        for q in range(n_queries):
            while True:
                try:
                    representation = model_stealing(path_to_png_file)
                    feature_representations[i, q] = representation
                except Exception as e:
                    print(f"Error querying {path_to_png_file}, {q}: {e}")
                else:
                    print(f'processed {path_to_png_file}, {q}')
                    break


        # Save progress every checkpoint_interval images
        if (i + 1) % checkpoint_interval == 0 or i == len(files) - 1:
            np.save(save_path, feature_representations)
            print(f"Checkpoint saved at image index {i}")

    return feature_representations








directory = './saved_images_sorted'
n_queries = 5

save_path = 'feature_representations_sorted.npy'
feature_representations = model_stealing_with_checkpoint(directory, n_queries, 100, save_path)


save_path = 'feature_representations_final.npy'
np.save(save_path, feature_representations)
print(f"Feature representations saved to: {save_path}")
