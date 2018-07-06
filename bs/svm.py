import cv2
import os

import matplotlib
import numpy as np

matplotlib.interactive(True)

kernel = np.ones((4, 4), np.float32) / 16
TARGET_FRAME_NUM = 50


def random_pack_video(video, num_frms):
    num = num_frms - video.shape[0]
    if num == 1:
        return np.concatenate([video, video[-1:]], axis=0)
    else:
        thres = np.random.randint(num - 1) + 1
        pre = np.tile(np.zeros_like(video[0]), (thres, 1))
        suf = np.tile(np.zeros_like(video[0]), (num_frms - len(video) - thres, 1))
        return np.concatenate([pre, video, suf], axis=0)


def random_trim_video(video, num_frms):
    num = len(video) - num_frms
    thres = np.random.randint(num + 1)
    return video[thres:num_frms + thres]


def prepare_feature(data_path='../data/'):
    feats_label = []
    with open('../data/ntu-videos.txt', 'r') as infile:
        lines = infile.readlines()

    for v_count, video in enumerate(lines):
        video = video.strip('\n')
        print('{}/{}, {}'.format(v_count + 1, len(lines), video))
        x_feats = []
        y_feats = []

        cap = cv2.VideoCapture(os.path.join(data_path, video))
        fgbg = cv2.createBackgroundSubtractorMOG2(history=2000, varThreshold=60)
        ret, frame1 = cap.read()
        SCALE = 16
        down_frame1 = cv2.resize(frame1, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
        prvs = cv2.cvtColor(down_frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(down_frame1)
        hsv[..., 1] = 255

        while ret:
            ret, frame2 = cap.read()
            if not ret:
                break

            smoothed = cv2.filter2D(frame2, -1, kernel)
            fgmask = fgbg.apply(smoothed)
            ret, fgmask = cv2.threshold(fgmask, 127, 255, cv2.THRESH_TOZERO)

            opening = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            Xs = np.where(opening > 100)[0]
            Ys = np.where(opening > 100)[1]
            if not np.all(Xs == 0) and not np.all(Ys == 0):
                left_new = np.min(Xs)
                right_new = np.max(Xs)
                top_new = np.min(Ys)
                bott_new = np.max(Ys)

                foreground = np.ones_like(opening) * 255
                foreground[left_new:right_new, top_new:bott_new] = opening[left_new:right_new, top_new:bott_new]
            else:
                foreground = np.ones_like(opening) * 255
            down_frame2 = cv2.resize(foreground, (0, 0), fx=1.0 / SCALE, fy=1.0 / SCALE)
            next = down_frame2
            if np.max(opening) != np.min(opening):
                flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                dx = flow[..., 0]
                dy = flow[..., 1]

                l, r = int(left_new / SCALE), int(right_new / SCALE)
                t, b = int(top_new / SCALE), int(bott_new / SCALE)
                delta_x = np.zeros_like(dx)
                delta_y = np.zeros_like(dy)
                delta_x[l:r, t:b] = dx[l:r, t:b]
                delta_y[l:r, t:b] = dy[l:r, t:b]

                x_feats.append(delta_x.flatten())
                y_feats.append(delta_y.flatten())

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            prvs = next
            ret, frame2 = cap.read()

        x_feats = np.vstack(x_feats)
        y_feats = np.vstack(y_feats)
        xy_feats = np.hstack((x_feats, y_feats))
        if xy_feats.shape[0] < TARGET_FRAME_NUM:
            xy_feats = random_pack_video(xy_feats, TARGET_FRAME_NUM)
        elif xy_feats.shape[0] > TARGET_FRAME_NUM:
            xy_feats = random_trim_video(xy_feats, TARGET_FRAME_NUM)

        if 'A043' in video:
            label = 'fall'
        else:
            label = 'other'
        feats_label.append((xy_feats, label))

        cap.release()
    np.save('../data/feats-for-svm', feats_label)
    return feats_label


def dim_reduction(feats):
    X = []
    L = []
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split

    assert feats[0][0].shape[0] == TARGET_FRAME_NUM, 'length of feature does not meet requirement'
    count = 0
    for v_feat, v_label in feats:
        count += 1
        print(count, len(feats))
        # pca = PCA(n_components=2000)
        # pca.fit(v_feat)
        # X.append(pca.transform(v_feat))
        X.append(v_feat)
        L.append(v_label)

    X_train, X_test, L_train, L_test = train_test_split(X, L, test_size=0.4, random_state=0)
    np.save('../data/split', (X_train, X_test, L_train, L_test))

    return X_train, X_test, L_train, L_test


def do_svm(X_train, X_test, L_train, L_test):
    from sklearn.svm import SVC

    for i in range(len(X_train)):
        X_train[i] = X_train[i].flatten()
    for i in range(len(X_test)):
        X_test[i] = X_test[i].flatten()
    clf = SVC()
    clf.fit(X_train, L_train)
    print('score: ', clf.score(X_test, L_test))
    print('pred label: ', clf.predict(X_test))


if __name__ == '__main__':
    # prepare_feature('/home/xiaoyu/Documents/action/datasets/ntu/test_videos')
    # feats = np.load('../data/feats-for-svm.npy')
    # dim_reduction(feats)
    X_train, X_test, L_train, L_test = np.load('../data/split.npy')
    do_svm(X_train, X_test, L_train, L_test)
