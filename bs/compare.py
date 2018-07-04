import cv2
import os
import numpy as np
from skimage import morphology


class OpencvBS(object):
    def __init__(self):
        self.kernel = np.ones((3, 3), np.float32) / 9
        self.data_path = '../data/'
        self.cap = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
        self.det = []
        self.count = 0

    def run(self):
        for video in os.listdir(self.data_path):
            if video.endswith('.mp4'):
                continue
            self.cap = cv2.VideoCapture(self.data_path + video)
            cv2.namedWindow('frame')
            cv2.namedWindow('fgmask')
            cv2.moveWindow('fgmask', 960, 0)
            ret, frame = self.cap.read()
            while ret:
                self.count += 1
                frame = cv2.resize(frame, (960, 540))
                smoothed = cv2.filter2D(frame, -1, self.kernel)
                fgmask = self.fgbg.apply(smoothed)

                # fgmask = fgbg.apply(frame)

                ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_TOZERO)

                opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self.kernel)
                opening = morphology.remove_small_objects(opening, min_size=400, connectivity=2)
                # opening[np.where(opening > 0)] = 255

                X = np.where(opening > 100)[0]
                Y = np.where(opening > 100)[1]

                if not np.all(X == 0) and not np.all(Y == 0):
                    left_new = np.min(X)
                    right_new = np.max(X)
                    top_new = np.min(Y)
                    bott_new = np.max(Y)

                    self.det.append([self.count, 1, left_new, top_new, right_new - left_new, bott_new - top_new,
                                     -1, -1, -1, -1])

                    cv2.rectangle(opening, (top_new, left_new), (bott_new, right_new), (255, 0, 255), 2)
                    cv2.rectangle(frame, (top_new, left_new), (bott_new, right_new), (0, 0, 255), 2)

                cv2.imshow('fgmask', opening)
                cv2.imshow('frame', frame)
                # cv2.imshow('smooth', smoothed)

                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break

                ret, frame = self.cap.read()

            self.cap.release()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    bsObj = OpencvBS()
    bsObj.run()
    print(bsObj.det)
