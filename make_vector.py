# import glob
# import re
#
# import cv2
# import numpy as np
#
# from db_config import get_connection
#
# NUM_FEATURES = 100
# NOR_X = 500
# NOR_Y = 500
# PAD_SIZE = 16000
#
#
import re


def preprocess_filename(filename, max_length=None):
    # 공백 및 특수 문자 제거
    cleaned_filename = re.sub(r'[^\w\s.-]', '', filename)

    # 파일 이름 길이 제한
    if max_length is not None and len(cleaned_filename) > max_length:
        cleaned_filename = cleaned_filename[:max_length]

    # 파일 이름 인코딩
    encoded_filename = cleaned_filename.encode('utf-8')

    return encoded_filename
#
#
# def reduce_dimension(features, target_dimension):
#     # SIFT 특징 벡터를 행렬로 변환
#     features_matrix = np.array([np.float64(feature) for feature in features])
#
#     # 평균 제거
#     mean = np.mean(features_matrix, axis=0)
#     features_matrix -= mean
#
#     # 공분산 행렬 계산
#     covariance_matrix = np.cov(features_matrix.T)
#
#     # 고유값과 고유벡터 계산
#     eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
#
#     # 고유값을 내림차순으로 정렬하고 상위 주성분 선택
#     idx = eigenvalues.argsort()[::-1]
#     eigenvectors = eigenvectors[:, idx][:, :target_dimension]
#
#     # 특징 벡터를 주성분에 투영하여 차원 축소
#     reduced_features = np.dot(features_matrix, eigenvectors)
#
#     reduced_features = np.round(reduced_features).astype(int)
#
#     return reduced_features
#
#
# def get_sift(n_features, n_octave_layers, contrast_threshold, edge_threshold, sigma):
#     return cv2.xfeatures2d.SIFT_create(nfeatures=n_features,
#                                        nOctaveLayers=n_octave_layers,
#                                        contrastThreshold=contrast_threshold,
#                                        edgeThreshold=edge_threshold,
#                                        sigma=sigma)
#
#
# def get_grey_images():
#     path_data = "test_dataset"
#     ext = ['png', 'jpg', 'jpeg']
#     for e in ext:
#         for image_path in glob.glob(path_data + "/**/*." + e, recursive=True):
#             image = cv2.imread(image_path)
#             resized_image = cv2.resize(image, (NOR_X, NOR_Y), interpolation=cv2.INTER_LINEAR)
#             if resized_image.ndim == 2:
#                 gray_image = resized_image
#             else:
#                 gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
#             yield image_path, gray_image
# import glob


def get_images_path():
    path_data = "dataset"
    ext = ['png', 'jpg', 'jpeg']
    for e in ext:
        for image_path in glob.glob(path_data + "/**/*." + e, recursive=True):
            yield image_path

# SIFT 이용
# if __name__ == '__main__':
#     # DB CONFIG
#     conn = get_connection()
#     cursor = conn.cursor()
#
#     sift = get_sift(n_features=NUM_FEATURES, n_octave_layers=3, contrast_threshold=0.04, edge_threshold=10, sigma=1.6)
#     index_dict = {}
#     for image_path, image in get_grey_images():
#         image_path = preprocess_filename(image_path).decode()
#         kp, des = sift.detectAndCompute(image, None)
#         sift_feature = np.array(des).flatten().tolist()
#         image_dict = {'file_path': image_path, 'feature_vector': sift_feature}
#         index_dict.update(image_dict)
#         sift_feature = sift_feature + [0] * (PAD_SIZE - len(sift_feature))
#
#         # # 1. 데이터베이스에 저장하는 코드
#         # query = f"""
#         # INSERT INTO image_vectors (img_path, embedding) VALUES('{image_path}', '{sift_feature}')
#         # """
#         # cursor.execute(query)
#
#         # 2. 데이터베이스에 저장된 내용을 검색하는 코드
#         query = f"""
#                     SELECT img_path FROM image_vectors ORDER BY embedding <-> '{sift_feature}' LIMIT 5;
#                     """
#         cursor.execute(query)
#         result = cursor.fetchall()
#         print(result)
#
#     conn.commit()


# 인공지능 이용

import numpy as np
from keras.applications import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model
from keras_preprocessing import image
from db_config import get_connection


def vectorize(filepath, model):
    layer_name = 'avg_pool'
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    img = image.load_img(filepath, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    intermediate_output = intermediate_layer_model.predict(x)
    return intermediate_output[0]


if __name__ == '__main__':
    # DB CONFIG
    model = InceptionResNetV2()

    conn = get_connection()
    cursor = conn.cursor()

    # # 데이터 삽입
    # for idx, image_path in enumerate(get_images_path()):
    #     img_vec = vectorize(image_path, model)
    #     img_vec = img_vec.flatten().tolist()
    #
    #     image_path = preprocess_filename(image_path).decode()
    #
    #     query = f"""
    #             INSERT INTO image_vectors (img_path, embedding) VALUES('{image_path}', '{img_vec}')
    #             """
    #     cursor.execute(query)
    #     print(idx + 1)
    # conn.commit()

    # 데이터 검색
    image_path = 'test_dataset/hotel_california.jpg'
    img_vec = vectorize(image_path, model)
    img_vec = img_vec.flatten().tolist()
    query = f"""
                    SELECT img_path FROM image_vectors ORDER BY embedding <-> '{img_vec}' LIMIT 1;
            """
    cursor.execute(query)
    result = cursor.fetchall()
    print(result)



