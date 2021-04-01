from video2tfrecord import convert_videos_to_tfrecord
from pathlib import Path
import os, glob
import colab_test
import cv2

import numpy as np

TVHI = Path(".") / "TV-HI"

VIDEOS = TVHI / "tv_human_interactions_videos"

FRAMES = VIDEOS / "frames"
if not colab_test.RUNNING_IN_COLAB and not FRAMES.exists():
    FRAMES.mkdir()

OPTICALFLOW = VIDEOS / "opticalflow"
if not colab_test.RUNNING_IN_COLAB and not OPTICALFLOW.exists():
    OPTICALFLOW.mkdir()


os.chdir(VIDEOS)
for file in glob.glob("*.avi"):

    video = os.path.basename(file)

    # OBTAIN FRAME IN THE MIDDLE
    cap = cv2.VideoCapture(video)
    # print('opened ' + filename)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(length)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(length / 2))
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    f1 = "frames/" + video.split(".")[0] + ".png"
    cv2.imwrite(f1, frame)

    cap.release()  # we close the video and we open it again cause it gives problems if i remember correctly

    # NOW TIME TO WORK ON THE BIG GUNS
    cap = cv2.VideoCapture(video)
    position = 0
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    nframes = 12  # the number of frames we want. arbitrary
    gap = int(length / nframes)
    i = 0
    while cap.isOpened():
        ret, frame1 = cap.read()
        if not ret:
            print("Can't receive frame for video " + video + ". Exiting ...")
            break
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        ret, frame2 = cap.read()
        if not ret:
            print("Can't receive frame for video " + video + ". Exiting ...")
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        f2 = "opticalflow/" + video.split(".")[0] + "_" + str(i) + ".png"
        cv2.imwrite(f2, rgb)
        print("frame " + str(i) + " processed in file:  " + f2)
        i = i + 1
        position = position + gap
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        if (position > length) or (i == nframes):
            break

    cap.release()
    cv2.destroyAllWindows()


print("we made it")
