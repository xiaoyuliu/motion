import cv2
import os
from skimage import morphology
import numpy as np

kernel = np.ones((2, 2), np.uint8)


def run():
    for video in os.listdir('../data'):
        if video != 'S001C002P006R001A043_rgb.avi':
            continue
        print(video)
        cap = cv2.VideoCapture('../data/{}'.format(video))
        fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
        cv2.namedWindow('frame')
        cv2.namedWindow('fgmask')
        cv2.moveWindow('fgmask', 960, 0)
        ret, frame = cap.read()
        while ret:
            frame = cv2.resize(frame, (960, 540))

            kernel = np.ones((2, 2), np.float32) / 4
            smoothed = cv2.filter2D(frame, -1, kernel)
            fgmask = fgbg.apply(smoothed)

            # fgmask = fgbg.apply(frame)

            ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_TOZERO)

            opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            opening = morphology.remove_small_objects(opening, min_size=400, connectivity=2)
            # opening[np.where(opening > 0)] = 255

            X = np.where(opening > 100)[0]
            Y = np.where(opening > 100)[1]

            if not np.all(X == 0) and not np.all(Y == 0):
                left_new = np.min(X)
                right_new = np.max(X)
                top_new = np.min(Y)
                bott_new = np.max(Y)

                cv2.rectangle(opening, (top_new, left_new), (bott_new, right_new), (255, 0, 255), 2)
                cv2.rectangle(frame, (top_new, left_new), (bott_new, right_new), (0, 0, 255), 2)

            cv2.imshow('fgmask', opening)
            cv2.imshow('frame', frame)
            # cv2.imshow('smooth', smoothed)

            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
