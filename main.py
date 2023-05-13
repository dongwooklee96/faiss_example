from src.searh import *

if __name__ == '__main__':
    path_image = 'dataset/Black_Sabbath_-_Paranoid.jpeg'
    similar = get_image_similar(path_image)
    print(similar)

