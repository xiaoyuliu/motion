import os

data_path = '/home/xiaoyu/Documents/action/datasets/ntu/test_videos'
videos = os.listdir(data_path)
fall_video = []
nonfall_video = []
for video in videos:
    if 'A043' in video:
        fall_video.append(video)
    else:
        nonfall_video.append(video)

import random

random.shuffle(nonfall_video)
select_nonfall = nonfall_video[:len(fall_video)]
all = fall_video + select_nonfall
random.shuffle(all)

with open('../data/ntu-videos.txt', 'w') as outfile:
    for v in all:
        outfile.write(v + '\n')
