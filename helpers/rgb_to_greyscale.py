import cv2
import os


PATH_TO_DATASET = "/raid/ankit/srinath/color_ARF/data/llff/fern/_images"
PATH_TO_DUMP = "/raid/ankit/srinath/color_ARF/data/llff/fern/images"

if not os.path.exists(PATH_TO_DUMP):
    os.mkdir(PATH_TO_DUMP)
images = os.listdir(PATH_TO_DATASET)
for image in images:
    im_gray = cv2.imread(os.path.join(PATH_TO_DATASET, image), cv2.IMREAD_GRAYSCALE)
    print(os.path.join(PATH_TO_DUMP, f"grey_{image}"))
    cv2.imwrite(os.path.join(PATH_TO_DUMP, f"grey_{image}"), im_gray)