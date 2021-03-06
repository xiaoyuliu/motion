import cv2
import os

import matplotlib
import numpy as np

matplotlib.interactive(True)
import matplotlib.pyplot as plt

kernel = np.ones((4, 4), np.float32) / 16

for video in os.listdir('../data'):
    if not video.endswith('.avi'):
        continue
    cap = cv2.VideoCapture("../data/{}".format(video))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
    ret, frame1 = cap.read()
    SCALE = 16
    down_frame1 = cv2.resize(frame1, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
    prvs = cv2.cvtColor(down_frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(down_frame1)
    hsv[..., 1] = 255

    IMAGE_H, IMAGE_W = down_frame1.shape[:2]
    fig = plt.figure(figsize=(IMAGE_W / 10.0, IMAGE_H / 10.0))
    ax = fig.add_subplot(1, 1, 1)
    while ret:
        ret, frame2 = cap.read()
        if not ret:
            break

        smoothed = cv2.filter2D(frame2, -1, kernel)
        fgmask = fgbg.apply(smoothed)

        # fgmask = fgbg.apply(frame)

        ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_TOZERO)

        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        Xs = np.where(opening > 100)[0]
        Ys = np.where(opening > 100)[1]
        if not np.all(Xs == 0) and not np.all(Ys == 0):
            left_new = np.min(Xs)
            right_new = np.max(Xs)
            top_new = np.min(Ys)
            bott_new = np.max(Ys)

            cv2.rectangle(frame2, (top_new, left_new), (bott_new, right_new), (255, 0, 255), 2)
            foreground = np.ones_like(opening) * 255
            foreground[left_new:right_new, top_new:bott_new] = opening[left_new:right_new, top_new:bott_new]
        else:
            foreground = np.ones_like(opening) * 255
        down_frame2 = cv2.resize(foreground, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
        next = down_frame2
        if np.max(opening) != np.min(opening):
            import time
            start = time.time()
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            # print('time per frame: ', time.time() - start)
            dx = flow[..., 0]
            dy = flow[..., 1]

            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            new_frame = np.zeros_like(frame1)

            X = range(0, IMAGE_W * SCALE, SCALE)
            Y = range(0, IMAGE_H * SCALE, SCALE)
            l, r = int(left_new / SCALE), int(right_new / SCALE)
            t, b = int(top_new / SCALE), int(bott_new / SCALE)
            delta_x = np.zeros_like(dx)
            delta_y = np.zeros_like(dy)
            delta_x[l:r, t:b] = dx[l:r, t:b]
            delta_y[l:r, t:b] = dy[l:r, t:b]
            U = delta_x
            V = -delta_y
            ax.clear()
            start = time.time()
            ax.quiver(X, Y, U, V, np.arctan2(V, U), scale=SCALE * SCALE, linewidth=.001, width=0.001)
            print('time per frame: ', time.time() - start)
        ax.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        # ax.imshow(foreground)
        plt.show()
        plt.pause(0.00000001)
        # bgr[np.where(bgr > 0)] = 127
        # cv2.imshow('frame', frame2)
        # k = cv2.waitKey(0) & 0xff
        # if k == 27:
        #     break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame2)
        prvs = next
        ret, frame2 = cap.read()
    plt.close()
    cap.release()
    cv2.destroyAllWindows()
