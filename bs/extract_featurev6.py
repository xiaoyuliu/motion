import argparse
import glob

import numpy as np

num_frames = 50
method = 'falling'


def get_mean_scale(pose_tensor, x_mean, y_mean):
    denorm = np.sum(pose_tensor[:, 2, :])
    # x_se = np.sum( np.square(pose_tensor[:,0,:]-x_mean) * pose_tensor[:,2,:])
    y_se = np.sum(np.square(pose_tensor[:, 1, :] - y_mean) * pose_tensor[:, 2, :])
    scale = np.maximum(np.sqrt(y_se / np.maximum(denorm, 1e-1)), 1e-3)
    return


def get_mean_scale2(pose_tensor, x_mean, y_mean):
    d1 = pose_tensor[:, 0:2, 1] - pose_tensor[:, 0:2, 11]
    d2 = pose_tensor[:, 0:2, 1] - pose_tensor[:, 0:2, 8]
    length1 = np.sqrt(np.sum(d1 * d1, axis=1))
    length2 = np.sqrt(np.sum(d2 * d2, axis=1))
    weights1 = np.minimum(pose_tensor[:, 2, 1], pose_tensor[:, 2, 11])
    weights2 = np.minimum(pose_tensor[:, 2, 1], pose_tensor[:, 2, 8])
    scale_m = (np.sum(weights1 * length1) + np.sum(weights2 * length2)) / (np.sum(weights2) + np.sum(weights1) + 1e-3)
    if scale_m < 1e-3:
        scale_m = 1000000.0
    return scale_m


def get_center(pose_tensor):
    denorm = np.sum(pose_tensor[:, 2, :])
    x_mean = np.sum(pose_tensor[:, 0, :] * pose_tensor[:, 2, :]) / np.maximum(denorm, 1e-3)
    y_mean = np.sum(pose_tensor[:, 1, :] * pose_tensor[:, 2, :]) / np.maximum(denorm, 1e-3)
    return x_mean, y_mean


def normalize_coords(pose_tensor, x_mean, y_mean, m_scale):
    x_t = (pose_tensor[:, 0:1, :] - x_mean) / m_scale
    y_t = (pose_tensor[:, 1:2, :] - y_mean) / m_scale
    return np.concatenate([x_t, y_t, pose_tensor[:, 2:3, :]], axis=1)


def normalize(pose_tensor):
    x_mean, y_mean = get_center(pose_tensor)
    m_scale = get_mean_scale2(pose_tensor, x_mean, y_mean)
    # print(x_mean, y_mean,m_scale)
    return normalize_coords(pose_tensor, x_mean, y_mean, m_scale)


pairs = [[1, 8], [1, 11], [8, 10],
         [11, 13], [2, 4], [5, 7],
         [0, 10], [0, 13],
         [14, 10], [14, 13],
         [15, 10], [15, 13],
         [16, 10], [16, 13],
         [17, 10], [17, 13],
         [1, 10], [1, 13]]
ranges = [5, 10, 15, 20, 30, 40, num_frames - 1]


def extract_feature(pose_tensor):
    # jointwise features  126 dims
    speeds = []
    for rng in ranges:
        speed = pose_tensor[rng:num_frames, 1, :] - pose_tensor[0:num_frames - rng, 1, :]
        scores = np.minimum(pose_tensor[rng:num_frames, 2, :], pose_tensor[0:num_frames - rng, 2, :])
        index = np.argmax(np.absolute(speed * scores), axis=0)
        # print(index)
        speeds.extend([speed[ind, joint] for (ind, joint) in zip(index, range(18))])
        # print(speeds[0].shape)
    speeds = np.asarray(speeds)
    # print(speeds.shape)

    # skeletal features  
    skeletal_movements = []
    for rng in ranges:
        for pair in pairs:
            start = pair[0]
            end = pair[1]
            sklt1 = pose_tensor[rng: num_frames, 0:2, end] - pose_tensor[rng: num_frames, 0:2, start]
            score1 = np.minimum(pose_tensor[rng: num_frames, 2, end], pose_tensor[rng: num_frames, 2, start])

            sklt0 = pose_tensor[0:num_frames - rng, 0:2, end] - pose_tensor[0:num_frames - rng, 0:2, start]
            score0 = np.minimum(pose_tensor[0:num_frames - rng, 2, end], pose_tensor[0:num_frames - rng, 2, start])

            score = np.minimum(score0, score1)
            index_x = np.argmax(np.absolute((sklt1 - sklt0)[:, 0]) * score, axis=0)
            x_movement = np.absolute((sklt1 - sklt0)[index_x, 0])
            index_y = np.argmax(np.absolute((sklt1 - sklt0)[:, 1] * score), axis=0)
            y_movement = (sklt1 - sklt0)[index_y, 1]
            skeletal_movements.append(x_movement)
            skeletal_movements.append(y_movement)
    return np.concatenate([speeds, np.asarray(skeletal_movements)], axis=0)


def random_pack_video(video, num_frms):
    num = num_frms - len(video)
    if num == 1:
        return np.concatenate([video, video[-1:]], axis=0)
    else:
        thres = np.random.randint(num - 1) + 1
        pre = np.tile(video[0:1], (thres, 1, 1))
        suf = np.tile(video[-1:], (num_frms - len(video) - thres, 1, 1))
        return np.concatenate([pre, video, suf], axis=0)


def random_trim_video(video, num_frms):
    num = len(video) - num_frms
    thres = np.random.randint(num + 1)
    return video[thres:num_frms + thres]


def get_features(video):
    if len(video) < num_frames:
        video = random_pack_video(video, num_frames)
    elif len(video) > num_frames:
        video = random_trim_video(video, num_frames)

    video = normalize(video)
    # print('after normalize', video)
    features = extract_feature(video)
    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Visualize the skeletons')
    parser.add_argument(
        '--data_path', default='data/fall_trim/*.npy')
    dst_path = 'data/features'
    arg = parser.parse_args()

    paths = glob.glob(arg.data_path)

    # print(len(paths))
    for path in paths:
        video = np.load(path)
        video_name = path.split('/')[-1]
        label = int(video_name.split('.')[-2])

        # np.save(dst_path+'/'+'features_'+str(num_frames) + '_' + method +"_"+video_name, features)
