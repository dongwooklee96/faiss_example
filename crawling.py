import requests
from bs4 import BeautifulSoup
import urllib
import os

# Discogs 웹 페이지 URL
url = 'https://www.discogs.com/ko/search/?genre_exact=Rock&sort=have%2Cdesc&ev=gs_mc'

# 웹 페이지 내용 가져오기
response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
html_content = response.text

# BeautifulSoup 객체 생성
soup = BeautifulSoup(html_content, 'html.parser')

# 앨범 이미지 태그(<img>)를 찾아서 이미지 URL 추출
albums = []
album_list_ul = soup.find('ul', id='search_results')
album_li_tags = album_list_ul.find_all('li')

for album_tag in album_li_tags:
    img_tag = album_tag.find('img')
    if img_tag and 'data-src' in img_tag.attrs:
        image_url = img_tag['data-src']
        albums.append({
            'title': img_tag['alt'],
            'url': image_url
        })

# 이미지를 저장할 디렉토리 생성
save_dir = 'dataset'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 이미지 다운로드
for album in albums:
    image_url = album.get('url')
    image_name = album.get('title').replace(' ', '_') + '.jpeg'
    save_path = os.path.join(save_dir, image_name)

    try:
        urllib.request.urlretrieve(image_url, save_path)
        print(f'Saved image: {save_path}')
    except Exception as e:
        print(f'Failed to save image: {image_url}')
        print(e)
