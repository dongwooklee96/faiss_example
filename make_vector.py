import glob

import cv2
import numpy as np

from db_config import get_connection
from psycopg2.extras import execute_values


NUM_FEATURES = 100
NOR_X = 512
NOR_Y = 384

def get_sift(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma):
    return cv2.xfeatures2d.SIFT_create(nfeatures=n_features,
                                       nOctaveLayers=n_octave_layers,
                                       contrastThreshold=contrast_threshold,
                                       edgeThreshold=edge_threshold,
                                       sigma=sigma)


def get_grey_images():
    path_data = "dataset"
    ext = ['png', 'jpg', 'jpeg']
    for e in ext:
        for image_path in glob.glob(path_data + "/**/*." + e, recursive=True):
            image = cv2.imread(image_path)
            resized_image = cv2.resize(image, (NOR_X, NOR_Y))
            if resized_image.ndim == 2:
                gray_image = resized_image
            else:
                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
            yield image_path, gray_image


if __name__ == '__main__':
    # DB CONFIG
    conn = get_connection()
    cursor = conn.cursor()
    ids_count = 0
    features = np.matrix([])

    sift = get_sift(n_features=NUM_FEATURES, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6)
    index_dict = {}
    for image_path, image in get_grey_images():
        kp, des = sift.detectAndCompute(image, None)
        if des is not None and len(des) > 0:
            sift_feature = np.matrix(des)
            image_dict = {'file_path': image_path, 'feature_vector': sift_feature}
            index_dict.update(image_dict)
            ids_list = np.linspace(ids_count, ids_count, num=sift_feature.shape[0], dtype="int64")
            ids_count += 1

            if features.any():
                features = np.vstack((features, sift_feature))
                ids = np.hstack((ids, ids_list))
            else:
                features = sift_feature
                ids = ids_list
            if ids_count % 500 == 499:
                features = np.matrix([])

    # Convert feature vectors to string representation for pg_vector
    query = "INSERT INTO image_vectors (file_path, feature_vector) VALUES %s;"
    execute_values(cursor, query, index_dict)

    conn.commit()
