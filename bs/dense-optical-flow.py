import cv2
import os
import numpy as np
import matplotlib
from skimage import morphology

matplotlib.interactive(True)
import matplotlib.pyplot as plt

kernel = np.ones((2, 2), np.float32) / 4

for video in os.listdir('../data'):
    if video.endswith('.mp4'):
        continue
    cap = cv2.VideoCapture("../data/{}".format(video))
    fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
    ret, frame1 = cap.read()
    SCALE = 16
    down_frame1 = cv2.resize(frame1, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
    prvs = cv2.cvtColor(down_frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(down_frame1)
    hsv[..., 1] = 255

    H, W = down_frame1.shape[:2]
    fig = plt.figure(figsize=(H, W))
    ax = fig.add_subplot(1, 1, 1)
    while ret:
        ret, frame2 = cap.read()
        if not ret:
            break

        smoothed = cv2.filter2D(frame2, -1, kernel)
        fgmask = fgbg.apply(smoothed)

        # fgmask = fgbg.apply(frame)

        ret, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_TOZERO)

        opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

        down_frame2 = cv2.resize(opening, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
        next = down_frame2
        if np.max(opening) != np.min(opening):
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx = flow[..., 0]
            dy = flow[..., 1]

            # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # hsv[..., 0] = ang * 180 / np.pi / 2
            # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            new_frame = np.zeros_like(frame1)

            X = range(0, W * SCALE, SCALE)
            Y = range(0, H * SCALE, SCALE)
            U = dx
            V = -dy
            ax.clear()
            ax.quiver(X, Y, U, V, np.arctan2(V, U), scale=SCALE*2, linewidth=.0001, width=0.001)
        ax.imshow(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        plt.show()
        plt.pause(0.00000001)
        # bgr[np.where(bgr > 0)] = 127
        # cv2.imshow('frame', frame2)
        k = cv2.waitKey(1) & 0xff
        if k == 27:
            break
        # elif k == ord('s'):
        #     cv2.imwrite('opticalfb.png', frame2)
        prvs = next
        ret, frame2 = cap.read()
    plt.close()
    cap.release()
    cv2.destroyAllWindows()
