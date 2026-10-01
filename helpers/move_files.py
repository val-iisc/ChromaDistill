import os
import shutil

CODE = '_c524'
SCENE = 'three_budda'
PATH = f"/raid/ankit/srinath/color_ARF/data/llff/{SCENE}/teacher_images_2/"
DUMP = f"/raid/ankit/srinath/color_ARF/data/llff/{SCENE}/gan_extra/"

if not os.path.isdir(PATH):
    os.makedirs(PATH)

if not os.path.isdir(DUMP):
    os.makedirs(DUMP)

images = [f for f in os.listdir(PATH) if not CODE in f]
rest = [f for f in os.listdir(PATH) if CODE in f]

for image in images:
    new_path = os.path.join(DUMP, image.replace(f'{CODE}.jpg', '.jpg'))
    path = os.path.join(PATH, image)
    shutil.move(path, new_path)

for image in rest:
    new_path = os.path.join(PATH, image.replace(f'{CODE}.jpg', '.jpg'))
    path = os.path.join(PATH, image)
    shutil.move(path, new_path)