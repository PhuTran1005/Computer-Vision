import json

from models.NearestNeighbor import NearestNeighbor
from data.load_images import CatDogDataset


with open('CS231\ImageClassifier\params\path_config.json', 'r') as f:
    paths = json.load(f)
TRAIN_IMAGES_PATH = paths['image_train_path']
TEST_IMAGES_PATH = paths['image_test_path']
IMAGE_SIZE = int(paths['image_size'])

if __name__ == '__main__':
    data = CatDogDataset()

    X_train, y_train, X_test = data.load_images(TRAIN_IMAGES_PATH, TEST_IMAGES_PATH, IMAGE_SIZE)

    model = NearestNeighbor()
    model.train(X_train, y_train)
    model.predict(X_test[:10])