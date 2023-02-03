import cv2
import os


PATH_TO_DATASET = "/data/srinath/ARF-svox2/data/llff/_fern/images"
PATH_TO_DUMP = "/data/srinath/ARF-svox2/data/llff/fern/images"

os.mkdir(PATH_TO_DUMP)
images = os.listdir(PATH_TO_DATASET)
for image in images:
    print(image)
    im_gray = cv2.imread(os.path.join(PATH_TO_DATASET, image), cv2.IMREAD_GRAYSCALE)
    print(os.path.join(PATH_TO_DUMP, f"grey_{image}"))
    cv2.imwrite(os.path.join(PATH_TO_DUMP, f"grey_{image}"), im_gray)