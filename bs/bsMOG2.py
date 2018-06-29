import cv2
import os

import numpy as np

import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

dbscan = cluster.DBSCAN(eps=400)

def run():
    for video in os.listdir('../data'):
        cap = cv2.VideoCapture('../data/{}'.format(video))
        fgbg = cv2.createBackgroundSubtractorMOG2()
        cv2.namedWindow('frame')
        cv2.namedWindow('fgmask')
        cv2.moveWindow('fgmask', 960, 0)
        ret, frame = cap.read()
        while ret:
            frame = cv2.resize(frame, (960, 540))
            fgmask = fgbg.apply(frame)

            img_h, img_w = frame.shape[:2]

            # dbscan.fit(fgmask)
            loc = np.where(fgmask > 0)
            pred = dbscan.fit_predict(np.array(loc).T)

            pred = np.array(pred) + 1

            counts = np.bincount(pred)
            max_cluster = np.argmax(counts)

            # fgmask[pred[np.where(pred == max_cluster)].T] = 255
            fgmask[pred[np.where(pred != max_cluster)].T] = 0

            cv2.imshow('fgmask', fgmask)
            cv2.imshow('frame', frame)


            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break

            ret, frame = cap.read()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    run()
