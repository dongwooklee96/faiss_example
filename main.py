from faiss_index.faiss_index import FaissIndex

if __name__ == '__main__':
    index = FaissIndex()
    path_image = 'test_dataset/download.jpg'

    image = index.search_by_image(image=path_image, k=100)
    print(image)

