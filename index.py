import glob

import cv2

from db_config import get_connection
from src.color_descriptor import ColorDescriptor

path_data = "dataset"

conn = get_connection()
cursor = conn.cursor()

cd = ColorDescriptor((8, 12, 3))

ext = ['png', 'jpg', 'jpeg']

for e in ext:
    for imagePath in glob.glob(path_data + "/**/*." + e, recursive=True):
        try:
            image = cv2.imread(imagePath)

            features = cd.describe(image)

            features = [str(f) for f in features]
            features = ",".join(features)

            # insert features to database
            cursor.execute(f"""
                    INSERT INTO image_descriptor (path, color_descriptor)
                    VALUES ($${imagePath}$$, $${features}$$)
                    ON CONFLICT DO NOTHING;            
                    """)
            conn.commit()
        except Exception as e:
            print(e)
            print(imagePath)



