import numpy as np
import cv2

import Image_evaluation
import parameter

# 获取干扰能，同时修复干扰能 2.1

# src = "296059"
# src = "1"
# src = "aa1"
# src = "noise"
# src_or = "circle"
# src = "8068"
# src = "a"
# src = "b"
# src = "mubiao"
# src = "8068"
#
# src = "8068raw2"
# # src = "a0.001"
# src = "a0.001"
# src = "noise"
src = "8068"
src = "a0.001"
src = "noise"

inpath = "D:\\out\\"
outpath = "D:\\out\\"

raw = cv2.imread(inpath + src + ".jpg")
raw2 = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
groudTruth = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

cv2.imwrite(outpath + src + "gray" + ".jpg", raw2)
# np.savetxt(outpath + src + "raw2.csv", raw2, fmt="%d", delimiter=',')

tag = np.zeros((raw2.shape[0], raw2.shape[1]))
energy = np.zeros((raw2.shape[0], raw2.shape[1]))

xxx = [0, -1, -1, -1, 0, +1, +1, +1]
yyy = [+1, +1, 0, -1, -1, -1, 0, +1]


# noisetag = cv2.imread(inpath + "noisetag" + ".jpg")
# noisetag = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)


# h = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# for i in range(1, raw2.shape[0] - 1):
#     for j in range(1, raw2.shape[1] - 1):
#         a, b = find_noise(raw2, i, j)
#         tag[a, b] += 1

# 计算个点区分度大小
def find_energy(raw2, i, j):
    l = []
    nums = []
    for k in range(8):
        l.append(abs(int(raw2[i, j]) - int(raw2[i + xxx[k], j + yyy[k]])))
        nums.append(int(raw2[i + xxx[k], j + yyy[k]]))
    arr = sorted(nums)
    l = sorted(l)
    # print(l)
    # if l[0]>100:
    #     print(l)
    if raw2[i, j] > arr[len(arr) - 1] or raw2[i, j] < arr[0]:
        return 2 * l[0], 1
    else:
        return l[0], 0


for i in range(1, raw2.shape[0] - 1):
    for j in range(1, raw2.shape[1] - 1):
        en, t = find_energy(raw2, i, j)
        # tag[i, j] = t
        energy[i, j] = en
        # print(en)
        # energy[i, j] = en

# num1 = 0
# for i in range(1, raw2.shape[0] - 1):
#     for j in range(1, raw2.shape[1] - 1):
#         num = 0
#         for k in range(8):
#             if energy[i, j] > energy[i + xxx[k], j + yyy[k]]:
#                 num += 1
#         if num == 8:
#             tag[i, j] = 255
#             num1+=1
# print(num1)

num2 = 0
num1 = 0
num3 = 0

# 导入对比GroudTruth
noise = np.loadtxt(inpath + "noisetag.csv", dtype=np.int, delimiter=",", encoding='utf-8')
# print(noise)
# print(noise.shape[0])
# print(noise.shape[1])
for i in range(0, noise.shape[0] - 1):
    for j in range(0, noise.shape[1] - 1):
        if noise[i,j] == 255:
            num3+=1
# print(num3)

# 通过本算法对噪声定位的准确程度
for i in range(1, noise.shape[0] - 1):
    for j in range(1, noise.shape[1] - 1):
        # num = 0
        l = []
        for k in range(8):
            # if energy[i, j] > energy[i + xxx[k], j + yyy[k]]:
            #     num += 1
            l.append(energy[i + xxx[k], j + yyy[k]])
        # if num == 8:
        #     tag[i, j] = 255
        l = sorted(l)
        # print(l)
        if energy[i, j] >= l[len(l)-1]:
            # tag[i,j] = 255
            num2+=1
            if noise[i,j]==255:
                num1+=1
            tag[i,j]=255
# print(num2)
cv2.imwrite(outpath + src + "energy" + ".jpg", tag)
np.savetxt(outpath + src + "energy.csv", tag, fmt="%d", delimiter=',')
np.savetxt(outpath + "tag.csv", tag, fmt="%d", delimiter=',')


# 存储当前结果对上一步没有影响的结果
no_effect_re = np.zeros((raw2.shape[0], raw2.shape[1]))

# 修复，找到标记噪声点周围与本点像素值最接近的值
for i in range(1, raw2.shape[0] - 1):
    for j in range(1, raw2.shape[1] - 1):
        no_effect_re[i,j] = raw2[i,j]
        if tag[i, j] == 255:
            max = 255
            t = 0
            for k in range(8):
                if abs(int(raw2[i, j]) - int(raw2[i + xxx[k], j + yyy[k]])) < max:
                    max = int(raw2[i, j]) - int(raw2[i + xxx[k], j + yyy[k]])
                    t = k
            # raw2[i, j] = raw2[i + xxx[t], j + yyy[t]]
            no_effect_re[i, j] = raw2[i + xxx[t], j + yyy[t]]

raw2 = no_effect_re


cv2.imwrite(outpath + src + "raw2" + ".jpg", raw2)
np.savetxt(outpath + src + "raw2.csv", raw2, fmt="%d", delimiter=',')

print(num1) # num1 修改正确的
print(num3)  # num3 总共的噪声个数
print(num1/num3)  # num1 修改正确的,总共的噪声个数


print(num2) # 一共修改的
print(num1/num2)  # num1 修改正确的,一个修改的

print(Image_evaluation.psnr2(groudTruth,raw2))

raw2 = cv2.imread(inpath + "noiseraw2" + ".jpg")
raw2 = cv2.cvtColor(raw2, cv2.COLOR_BGR2GRAY)
canny3 = cv2.Canny(raw2, parameter.th1,parameter.th2)
cv2.imwrite(outpath + "canny3" + ".jpg", canny3)


