from datetime import datetime

import cv2
import faiss
import numpy

from db_config import get_connection
from src.color_descriptor import ColorDescriptor


def get_total_dataset_from_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM image_descriptor;
    """)
    rows = cursor.fetchall()
    return rows


def get_dataset_vector():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(f"""
            select color_descriptor from image_descriptor order by id;
        """)
    rows = cursor.fetchall()
    list_vector = []
    for row in rows:
        features = [float(x) for x in row[0].split(',')]
        list_vector.append(features)

    return numpy.array(list_vector)


def get_feature_vector(path_image):
    # define color descriptor
    cd = ColorDescriptor((8, 12, 3))

    # create ndarray by vectors of image file
    query = cv2.imread(path_image)
    features = cd.describe(query)
    features_vector = [features]

    return numpy.array(features_vector).astype('float32')


def get_index_vector():
    dim = 1440
    index = faiss.IndexFlatL2(dim)
    index = faiss.read_index("train.index")
    total_dataset = get_total_dataset_from_db()
    vector_total = index.ntotal

    if len(total_dataset) > vector_total:
        for page in range(1, 100):
            page_size = 10000

            # find dimension vectors
            data_vector = get_dataset_vector()

            if len(data_vector) > 0:
                vector_total = vector_total + page_size
                data_vector = data_vector.reshape(data_vector.shape[0], -1).astype('float32')
                index.add(data_vector)
            else:
                break
        # write index file to disk
        faiss.write_index(index, "train.index")
    print(index.ntotal)
    return index


def search_vector(path_image, total_results):
    index = get_index_vector()
    features_vector_search = get_feature_vector(path_image)

    # search image by the feature vector
    D, I = index.search(features_vector_search, total_results)
    index.reset()
    return I


def get_image_similar(path_image):
    print(datetime.now())
    conn = get_connection()
    cursor = conn.cursor()

    total_results = 1
    result_vector = search_vector(path_image, total_results)

    query = ""
    for j in range(total_results):
        query += f"(select id, path from image_descriptor order by id LIMIT 1 offset {result_vector[0][j]})"
    cursor.execute(query)
    rows = cursor.fetchall()
    print(datetime.now())
    return rows
