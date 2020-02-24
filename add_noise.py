import numpy as np
import cv2
import parameter
import Image_evaluation
src_or = "135069"
src_or = "1"
src_or = "circle"
src_or = "a0.001"
src_or = "8068"
src_or = "t1"

inpath = "D:\\out\\"
outpath = "D:\\out\\"

raw_or = cv2.imread(inpath + src_or + ".jpg")
tag = np.zeros((raw_or.shape[0], raw_or.shape[1]))
raw2 = cv2.cvtColor(raw_or, cv2.COLOR_BGR2GRAY)

print(raw_or.shape[0])
print(raw_or.shape[1])

th1 = parameter.th1
th2 = parameter.th2

canny1=cv2.Canny(raw2,th1,th2)
cv2.imwrite(outpath + "canny1" + ".jpg", canny1)

def add_noise(img, snr):
    h = img.shape[0]
    w = img.shape[1]
    img1 = img.copy()
    sp = h * w  # 计算图像像素点个数
    NP = int(sp * (1 - snr))  # 计算图像椒盐噪声点个数
    for i in range(NP):
        randx = np.random.randint(1, h - 1)  # 生成一个 1 至 h-1 之间的随机整数
        randy = np.random.randint(1, w - 1)  # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random() <= 0.5:  # np.random.random()生成一个 0 至 1 之间的浮点数
            img1[randx, randy] = 0
        else:
            img1[randx, randy] = 255
        tag[randx, randy] = 255
    return img1


def gasuss_noise(image, mean=0, var=0.001):
    '''
        添加高斯噪声
        mean : 均值
        var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    # cv.imshow("gasuss", out)
    return out


def verify_gass_noise(image, var):
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            if abs(int(image[i, j]) - int(raw2[i, j])) > var:
                tag[i, j] = 255


# re = add_noise(raw_or, 0.995)

# 添加高斯噪声，并判断是否
num = 5
# re = gasuss_noise(raw_or, 0, num*num / pow(255, 2))
re = gasuss_noise(raw2, 0, num*num / pow(255, 2))
# re = cv2.cvtColor(re, cv2.COLOR_BGR2GRAY)
verify_gass_noise(re, num)

print(Image_evaluation.psnr2(re,raw2))

cv2.imwrite(outpath + "noise" + ".jpg", re)
cv2.imwrite(outpath + "noisetag" + ".jpg", tag)
np.savetxt(outpath + "noisetag.csv", tag, fmt="%d", delimiter=',')

canny2 = cv2.Canny(re,th1,th2)
cv2.imwrite(outpath + "canny2" + ".jpg", canny2)

# print(pow(255,2))
