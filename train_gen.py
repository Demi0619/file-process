f1 = open("train/train_2019_07_09_15_51_56_cam1_RAW.txt", 'wt')

N = 45824

for i in range(N):
    print("/home/louis/workspace/darknet/data/2019_07_09_15_51_56_cam1_RAW/frame" + "{:06d}".format(i+1) + ".jpg", file=f1)

f1.close()
