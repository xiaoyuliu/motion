import cv2
import os

import matplotlib
import numpy as np

matplotlib.interactive(True)
import matplotlib.pyplot as plt

kernel = np.ones((3, 3), np.float32) / 9
measure_y = []
predict_y = []

for video in os.listdir('../data/'):
    if not video.endswith('.avi'):
        continue
    print(video)
    if len(measure_y) > 0:
        plt.plot(measure_y, label='measure')
        plt.plot(predict_y, label='predict')
        plt.legend()
        plt.show()
        plt.clf()
    measure_y = []
    predict_y = []
    cap = cv2.VideoCapture('../data/' + video)
    cv2.namedWindow('frame')
    cv2.namedWindow('fgmask')
    cv2.moveWindow('fgmask', 960, 0)
    fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
    ret, frame = cap.read()

    current_mes, mes, last_mes, current_pre, last_pre = None, None, None, None, None

    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.003
    kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1

    while ret:
        frame = cv2.resize(frame, (960, 540))
        smoothed = cv2.filter2D(frame, -1, kernel)
        fgmask = fgbg.apply(smoothed)

        # fgmask = fgbg.apply(frame)

        ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_TOZERO)

        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        # opening = morphology.remove_small_objects(opening, min_size=400, connectivity=2)
        # opening[np.where(opening > 0)] = 255

        X = np.where(opening > 100)[0]
        Y = np.where(opening > 100)[1]

        if not np.all(X == 0) and not np.all(Y == 0):
            left_new = np.min(X)
            right_new = np.max(X)
            top_new = np.min(Y)
            bott_new = np.max(Y)

            center_x = (left_new + right_new) / 2.0
            center_y = (top_new + bott_new) / 2.0

            last_mes = current_mes = np.array((2, 1), np.float32)
            last_pre = current_pre = np.array((2, 1), np.float32)

            last_pre = current_pre
            last_mes = current_mes
            current_mes = np.array([[np.float32(center_y)], [np.float32(center_x)]])

            kalman.correct(current_mes)
            current_pre = kalman.predict()

            measure_y.append(int(current_mes[1]))
            predict_y.append(int(current_pre[1]))

            if np.all(last_mes == 0):
                continue
            lmx, lmy = last_mes[0], last_mes[1]
            lpx, lpy = last_pre[0], last_pre[1]
            cmx, cmy = current_mes[0], current_mes[1]
            cpx, cpy = current_pre[0], current_pre[1]
            cv2.line(frame, (lmx, lmy), (cmx, cmy), (0, 200, 0), thickness=3)
            cv2.line(frame, (lpx, lpy), (cpx, cpy), (0, 0, 200), thickness=3)

            cv2.rectangle(opening, (top_new, left_new), (bott_new, right_new), (255, 0, 255), 2)
            cv2.rectangle(frame, (top_new, left_new), (bott_new, right_new), (0, 0, 255), 2)

        cv2.imshow('fgmask', opening)
        cv2.imshow('frame', frame)
        # cv2.imshow('smooth', smoothed)

        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

        ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
