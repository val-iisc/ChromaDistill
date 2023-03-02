import cv2
import os

scenes = os.listdir("/raid/ankit/srinath/color_ARF/data/llff")

for scene in scenes: 

    PATH_TO_DATASET = f"/raid/ankit/srinath/color_ARF/data/llff/{scene}/images"
    PATH_TO_DUMP = f"/raid/ankit/srinath/color_ARF/data/llff/{scene}/images"

    if not os.path.exists(PATH_TO_DUMP):
        os.mkdir(PATH_TO_DUMP)
    images = os.listdir(PATH_TO_DATASET)
    for image in images:
        im_gray = cv2.imread(os.path.join(PATH_TO_DATASET, image), cv2.IMREAD_GRAYSCALE)
        im_gray = cv2.resize(im_gray, (1008, 752))
        print(os.path.join(PATH_TO_DUMP, f"{image}"))
        cv2.imwrite(os.path.join(PATH_TO_DUMP, f"{image}"), im_gray)