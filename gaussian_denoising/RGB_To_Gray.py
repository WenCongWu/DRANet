import os
import glob
import cv2


def Togray():
    files_source = glob.glob(os.path.join('trainsets', 'Flickr2K_train_HR', '*.png'))
    files_source.sort()
    for f in files_source:
        print(f)
        img = cv2.imread(f, 0)
        path = os.path.join('trainsets', 'Flickr2K_train_HR_gray', f[-15:-4] + '.png')
        cv2.imwrite(path, img)


if __name__ == '__main__':
    Togray()








