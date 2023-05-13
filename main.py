from src.searh import *

if __name__ == '__main__':
    origin_image = 'dataset/Black_Sabbath_-_Paranoid.jpeg'
    path_image = 'test_dataset/hotel_california.jpg'
    similar = get_image_similar(path_image)
    print(similar)

