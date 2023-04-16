import cv2
import os
from os.path import join, dirname, abspath
from icecream import ic

vid_dir = r"Private_Test/videos"
vid_to_img_dir = r'Private_Test/images_from_videos'
sub_dir_lst = os.listdir(vid_dir)

def creat_frames(vid_dir, sub_dir_lst, vid_to_img_dir):
    for sub_dir_name in sub_dir_lst:
        os.mkdir(join(vid_to_img_dir, sub_dir_name))
        vid_name_lst = os.listdir(join(vid_dir,sub_dir_name))
        for vid_name in  vid_name_lst:
            # os.mkdir(join(vid_to_img_dir, sub_dir_name, vid_name[:-4]))
            cap = cv2.VideoCapture(join(vid_dir, sub_dir_name, vid_name))
            i = 1
            while True:
                ret, frame = cap.read()
                if ret:
                    if os.path.exists(join(vid_to_img_dir, sub_dir_name, "frame_{}".format(i))) == False:
                        os.mkdir(join(vid_to_img_dir, sub_dir_name, "frame_{}".format(i)))
                    cv2.imwrite(join(vid_to_img_dir, sub_dir_name, "frame_{}".format(i), vid_name[:-4]  + ".jpg".format(i)), frame)
                    i = i + 1
                else: break
creat_frames(vid_dir, sub_dir_lst, vid_to_img_dir)