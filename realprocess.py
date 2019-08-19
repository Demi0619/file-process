import re
from collections import Counter

from itertools import zip_longest


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


f1 = open("summary/cam1_PRC_bus_stop.txt", 'r')
f2 = open("summary/cam1_RAW_bus_stop.txt", 'r')
f3 = open("data/coco.names", 'r')
f4 = open("summary/cam1_bus_stop.txt", 'wt')

list1 = []
list2 = []
list3 = []

for lines in f1:
    for word in lines.split("\n"):
        list1.append(word)
for lines in f2:
    for word in lines.split("\n"):
        list2.append(word)
for lines in f3:
    for word in lines.split("\n"):
        list3.append(word)
list3 = list(filter(None, list3))
# print(list3)

object_prc = []
object_raw = []
axis_prc = []
axis_raw = []
threshold = int(input('Please input a threshold:\t'))

for i in list3:
    # i = i.strip()
    # print(i)
    count_object_prc = 0
    count_frame_prc = 0
    for j in list1:
        match_frame_prc = re.search(r'\d+', j)
        if match_frame_prc:
            count_frame_prc = int(match_frame_prc.group())
            my_regex1 = r"\b(?=\w)" + re.escape(i) + r"\b(?!\w)"
            match_object_prc = re.findall(my_regex1, j)
            if match_object_prc:
                object_prc.append([int(s) for s in re.findall(r'\d+', j)])  # PRC list [frame, left, top, right, bottom]
    axis_prc = object_prc
    object_prc = []
    for k in list2:
        match_frame_raw = re.search(r'\d+', k)
        if match_frame_raw:
            count_frame_raw = int(match_frame_raw.group())
            my_regex2 = r"\b(?=\w)" + re.escape(i) + r"\b(?!\w)"
            match_object_raw = re.findall(my_regex2, k)
            if match_object_raw:
                object_raw.append([int(s) for s in re.findall(r'\d+', k)])  # RAW list [frame, left, top, right, bottom]
    axis_raw = object_raw
    object_raw = []

    sublist_prc = [item[0] for item in axis_prc]  # PRC list: column 1
    # frame_prc = max(sublist_prc)
    dic_prc = Counter(sublist_prc)
    dic_frame_prc = sorted(dic_prc.items())  # dictionary (column 1) of PRC list
    # dic_prc.clear()

    sublist_raw = [item[0] for item in axis_raw]  # RAW list: column 1
    # frame_raw = max(sublist_raw)
    dic_raw = Counter(sublist_raw)
    dic_frame_raw = sorted(dic_raw.items())  # dictionary (column 1) of RAW list
    # dic_raw.clear()

    index_p = 0  # PRC list index
    tmp_p = 0
    tmp_r = 0
    for x in range(len(dic_frame_prc)):  # PRC frame outer loop
        index_r = 0  # RAW list index
        match_frame = 0
        prc_inner = int(dic_frame_prc[x][1])
        total_detect = 0
        for y in range(len(dic_frame_raw)):  # RAW frame outer loop
            raw_inner = int(dic_frame_raw[y][1])
            if dic_frame_raw[y][0] == dic_frame_prc[x][0]:  # Both PRC and RAW detect the object
                match_frame += 1
                frame_overall = dic_frame_prc[x][0]
                total_detect = raw_inner
                diff = [[0 for x in range(raw_inner)] for y in range(prc_inner)]
                index_p = tmp_p
                index_r = tmp_r
                for m in range(prc_inner):  # PRC frame inner loop
                    row = []
                    index_p = tmp_p + m
                    for n in range(raw_inner):  # RAW frame inner loop
                        index_r = tmp_r + n
                        diff[m][n] = abs(axis_prc[index_p][1] - axis_raw[index_r][1])**2 + \
                                     abs(axis_prc[index_p][2] - axis_raw[index_r][2])**2 + \
                                     abs(axis_prc[index_p][3] - axis_raw[index_r][3])**2 + \
                                     abs(axis_prc[index_p][4] - axis_raw[index_r][4])**2
                        row.append(diff[m][n])
                    if min(row) > threshold:
                        total_detect += 1
                # else:
                #    match_frame = 0
                tmp_r = tmp_r + raw_inner
                tmp_p = tmp_p + prc_inner
                print("frame\t", frame_overall, "\tdetect\t", i, "\t", total_detect, "\ttimes", file=f4)
            elif x == 0 and dic_frame_raw[y][0] < dic_frame_prc[x][0]:  # Only RAW detect the object
                frame_overall = dic_frame_raw[y][0]
                total_detect = raw_inner
                print("frame\t", frame_overall, "\tdetect\t", i, "\t", total_detect, "\ttimes", file=f4)
            elif x == len(dic_frame_prc) - 1 and dic_frame_raw[y][0] > dic_frame_prc[x][
                0]:  # Only RAW detect the object
                frame_overall = dic_frame_raw[y][0]
                total_detect = raw_inner
                print("frame\t", frame_overall, "\tdetect\t", i, "\t", total_detect, "\ttimes", file=f4)
            elif x != 0 and x != len(dic_frame_prc) - 1:
                if (dic_frame_raw[y][0] > dic_frame_prc[x][0]) and (
                        dic_frame_raw[y][0] < dic_frame_prc[x + 1][0]):  # Only RAW detect the object
                    frame_overall = dic_frame_raw[y][0]
                    total_detect = raw_inner
                    print("frame\t", frame_overall, "\tdetect\t", i, "\t", total_detect, "\ttimes", file=f4)
        if match_frame == 0:  # Only PRC detect the object
            frame_overall = dic_frame_prc[x][0]
            total_detect = prc_inner
            print("frame\t", frame_overall, "\tdetect\t", i, "\t", total_detect, "\ttimes", file=f4)

f1.close()
f2.close()
f3.close()
f4.close()
