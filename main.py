import os
import numpy as np
import cv2
import csv
from glob import glob
import torch
import matplotlib.pyplot as plt
import kornia
from kornia_moons.feature import *
import kornia as K
import kornia.feature as KF
import gc
from shapely.geometry import Polygon


device = torch.device('cuda')
matcher = KF.LoFTR(pretrained=None)
matcher.load_state_dict(torch.load("loftr_outdoor.ckpt")['state_dict'])
matcher = matcher.to(device).eval()


def get_image(src):
    lst = os.listdir(src)
    test_samples = [os.path.join(src, ls) for ls in lst]
    return sorted(test_samples)

def FlattenMatrix(M, num_digits=8):
    '''Convenience function to write CSV files.'''
    
    return ' '.join([f'{v:.{num_digits}e}' for v in M.flatten()])


def load_torch_image(fname, device):
    img = cv2.imread(fname)
    scale = 840 / max(img.shape[0], img.shape[1]) 
    w = int(img.shape[1] * scale)
    h = int(img.shape[0] * scale)
    img = cv2.resize(img, (w, h))
    img = K.image_to_tensor(img, False).float() /255.
    img = K.color.bgr_to_rgb(img)
    return img.to(device)

def infer(test_samples, test_loader):
    import time
    point = {}
    load_time = []
    for i in range(len(test_samples)):
        point[i] = []
        
    for i, row in enumerate(test_loader):
    #     sample_id, batch_id, image_1_id, image_2_id = row
        path_image_1 = row[0]
        path_image_2 = row[1]
        # Load the images.
        st = time.time()

        image_1 = load_torch_image(path_image_1, device)
        image_2 = load_torch_image(path_image_2, device)
        print(image_1.shape)
        input_dict = {"image0": K.color.rgb_to_grayscale(image_1), 
                "image1": K.color.rgb_to_grayscale(image_2)}

        with torch.no_grad():
            correspondences = matcher(input_dict)
            
        mkpts0 = correspondences['keypoints0'].cpu().numpy()
        mkpts1 = correspondences['keypoints1'].cpu().numpy()
        
        if len(mkpts0) > 7:
            #F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.25, 0.9999, 100000)
            F, inliers = cv2.findFundamentalMat(mkpts0, mkpts1, cv2.USAC_MAGSAC, 0.1845, 0.999999, 220000)
            inliers = inliers > 0
            assert F.shape == (3, 3), 'Malformed F?'
        else:
            continue
        gc.collect()
        nd = time.time()    
        print("Running time: ", nd - st, " s")
        load_time.append(nd - st)
        point[test_samples.index(path_image_1)].append([mkpts0[a] for a in range(len(inliers)) if inliers[a]])
        point[test_samples.index(path_image_2)].append([mkpts1[a] for a in range(len(inliers)) if inliers[a]])

    return point, load_time

def build_convex(point0):
    from shapely.geometry import Point, MultiPoint
    import matplotlib.pyplot as plt
    points = [Point(a, b) for a, b in point0]
    
    multipoint = MultiPoint(points)
    convex_hull = multipoint.convex_hull
    x = convex_hull.exterior.coords.xy[0]
    y = convex_hull.exterior.coords.xy[1]
    xy = [(x[i], y[i]) for i in range(len(x))]
    
    rescale_xy = [[x[0] * 1920 / 840, x[1] * 1920 / 840] for x in xy]    
    return rescale_xy

# point0, point1 is convex
def cal_intersection(point):
    poly = []
    coords = []
    for i in range(len(point)):
        poly.append(Polygon(point[i]))
    for i in range(1, len(point)):
        # Determine if the two polygons intersect
        if poly[i].intersects(poly[0]):
            # Find the intersection area
            intersection = poly[0].intersection(poly[i])
            overlap_area = intersection.area

            # Get all the coordinate points of the intersection polygon
            lis = []
            for num in list(intersection.exterior.coords):
                coords.append([num[0], num[1]])
            poly[0] = Polygon(coords)
        else:
            print("The two polygons do not intersect.")    
            return [[0, 0], [1080, 0], [1080, 1920], [0, 1920]]
    return coords

def view_coors(img_path, vertices):
    # Read the image using plt.imread()
    image = plt.imread(img_path)
    # Create a figure and axis object
    fig, ax = plt.subplots()
    # Show the image on the axis
    ax.imshow(image)
    # Create a patch object for the polygon and add it to the axis
    polygon = plt.Polygon(vertices, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(polygon)
    # Display the image with the polygon
    plt.show()

def get_overlap_by_id(id, point):
    point_convex = []
    for i in range(len(point[id])):
        point_convex.append(build_convex(point[id][i]))
    if len(point[id]) <= 1:
        return point_convex
    return cal_intersection(point_convex)

def get_overlap_by_dir(path):
    test_samples = get_image(path)
    test_loader = []
    for i in range(len(test_samples)):
        for j in range(i+1, len(test_samples)):
            test_loader.append([test_samples[i], test_samples[j]])
            
    point, load_time = infer(test_samples= test_samples, test_loader=test_loader)
    overlap = {}
    for id in range(len(test_samples)):
        overlap[id] = get_overlap_by_id(id, point)
    return overlap, load_time


def check_move(video_source):
    # Open the video file

    cap = cv2.VideoCapture(video_source)

    # Set the frame rate of the video to 5 fps
    cap.set(cv2.CAP_PROP_FPS, 5)
    # Read the first frame
    ret, prev_frame = cap.read()

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Loop through the video frames
    while True:
        # Read the current frame
        ret, curr_frame = cap.read()
        # Check if the frame has a valid size
        if ret:
            # Show the frame

            curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            # Compute the absolute difference between the current frame and the previous frame
            frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)

            # Threshold the result to detect motion
            _, thresh = cv2.threshold(frame_diff, 200, 255, cv2.THRESH_BINARY)
            if sum(sum(thresh)) > 100000:
                return True 

            # Update the previous frame
            prev_frame_gray = curr_frame_gray.copy()
        else:
            break
    return False


if __name__ == '__main__':
    # src = 'data'
    # import time
    # st = time.time()
    # overlap = get_overlap_by_dir(src)
    # et = time.time()
    import time
    src = "Private_Test/images_from_videos"
    save_path = "Private_Test/Submit"
    scenes = os.listdir(src)
    for scene in scenes:
        t1 = time.time()
        move = False
        scene_check_path = os.path.join("Private_Test\\videos", scene)
        for cam in os.listdir(scene_check_path):
            move = check_move(os.path.join(scene_check_path, cam))
            if move == True:
                break
        time_check_move = time.time() - t1
        os.mkdir(os.path.join(save_path, scene))
        overlaps = []
        if move == True:
            for frame_num, frame in enumerate(os.listdir(os.path.join(src,scene))):
                src_ls = os.path.join(src, scene, frame)
                overlap, load_time = get_overlap_by_dir(src_ls)
                if frame_num == 0:
                    load_time[0] += time_check_move
                for i in overlap.keys():
                    f = open(f"Private_Test/Submit/{scene}/CAM_{i+1}.txt", "a")

                    ans = f"frame_{frame_num+1}.jpg, ("
                    if len(overlap.keys()) <= 2:
                        for items in overlap[i]:
                            for item in items:
                                ans +=  str(int(item[0])) + "," + str(int(item[1])) + ","
                    else:
                        for items in overlap[i]:
                            ans +=  str(int(items[0])) + "," + str(int(items[1])) + ","
                    ans = ans[:-1]
                    ans += "), " + str(load_time[0]) + "\n"
                    f.writelines(ans)
                    f.close()
        else:
            first_frame_dir = os.path.join(src,scene, os.listdir(os.path.join(src,scene))[0])
            overlap, load_time = get_overlap_by_dir(first_frame_dir)
            for i in overlap.keys():
                f = open(f"Private_Test/Submit/{scene}/CAM_{i+1}.txt", "a")

                ans = f"frame_{1}.jpg, ("
                if len(overlap.keys()) <= 2:
                    for items in overlap[i]:
                        for item in items:
                            ans +=  str(int(item[0])) + "," + str(int(item[1])) + ","
                else:
                    for items in overlap[i]:
                        ans +=  str(int(items[0])) + "," + str(int(items[1])) + ","
                ans = ans[:-1]
                ans += "), " + str(load_time[0]) + "\n"
                f.writelines(ans)
                for frame_num, frame in enumerate(os.listdir(os.path.join(src,scene))):
                    if frame_num ==0:
                        continue
                    ans = f"frame_{frame_num+1}.jpg, ("
                    if len(overlap.keys()) <= 2:
                        for items in overlap[i]:
                            for item in items:
                                ans +=  str(int(item[0])) + "," + str(int(item[1])) + ","
                    else:
                        for items in overlap[i]:
                            ans +=  str(int(items[0])) + "," + str(int(items[1])) + ","
                    ans = ans[:-1]
                    ans += "), " + str(time_check_move) + "\n"
                    f.writelines(ans)
                f.close()
                
                

                  













































