import numpy as np
from math import sqrt
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit
import matplotlib.gridspec as gridspec

rsm = fits.open('./data/RSM20240501T064251_0024_HA.fits')

central_spec = rsm[1].data[:, 1800:1900,950:1050].mean((1, 2))

def pearson(vector1, vector2):
    # should have len(vector1) == len(vector2)
    n = len(vector1)

    sum1 = sum(vector1)
    sum2 = sum(vector2)

    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])

    p_sum = sum([vector1[i] * vector2[i] for i in range(n)])

    num = p_sum - (sum1 * sum2 / n)
    den = sqrt((sum1_pow - pow(sum1, 2) / n) * (sum2_pow - pow(sum2, 2) / n))
    if den == 0:
        return 0.0
    return num / den

def polyfit_p(x, y, n, draw = 0):
    # return para: a0, a1, a2, ..., an for y = a0 + a1 * x + a2 * x ** 2 + ... + an * x ** n
    pfpara, pfcov = np.polyfit(x, y, n, cov = True)

    if draw != 0:
        plt.subplot(211)
        lx2 = x
        ly2 = []
        for i in range(len(lx2)):
            y2 = 0
            for j in range(len(pfpara)):
                y2 += pfpara[-j - 1] * lx2[i] ** j
            ly2.append(y2)
        plt.plot(x, y, 'ro', lx2, ly2, 'b', linewidth = 2)
        plt.grid()

        plt.subplot(212)
        l0 = []
        lr = []
        for i in range(len(lx2)):
            l0.append(0)
            lr.append(y[i] - ly2[i])
        plt.plot(x, l0, 'black', linewidth = 2)
        plt.scatter(x, lr, color = 'red')
        plt.grid()

    para = list(pfpara)
    para = para[::-1]

    mat_cov = []
    for i in range(len(pfcov)):
        mat_cov.append(pfcov[-i - 1][-i - 1])

    return para, mat_cov

def get_dplv(i,j):
    ij_spec = rsm[1].data[50:90,i,j]
    dict_c = {}
    for lag in range(-10, 11):
        ha_profilel = central_spec[lag + 50:lag + 90] # move the ha profile, but keep the length
        pr = pearson(ha_profilel, ij_spec)
        dict_c[-lag] = pr
    if(i%100==0 and j %100 ==0):
        print(i,' ',j,'\n')
    # use cubic fit to get the lag of the largest pearson correlation
    lag_max = max(dict_c, key = lambda x:dict_c[x])
    l_lag = [lag_max - k for k in range(-2, 3)]
    l_pr=[]
    for k in l_lag:
        if k<=-11 or k>=11:
            return 0
        else:
            l_pr.append(dict_c[k])
    para_cf, pcov_cf = polyfit_p(l_lag, l_pr, 3)
    a0, a1, a2, a3 = para_cf[0], para_cf[1], para_cf[2], para_cf[3] # in format of a0 + a1 * x  + a2 * x ** 2 + a3 * x ** 3
    s0, s1, s2, s3 = pcov_cf[0], pcov_cf[1], pcov_cf[2], pcov_cf[3]

    dcf01 = (-2 * a2 + sqrt((2 * a2) ** 2 - 12 * a1 * a3)) / (6 * a3)
    dcf02 = (-2 * a2 - sqrt((2 * a2) ** 2 - 12 * a1 * a3)) / (6 * a3)

    d2cf1 = 6 * a3 * dcf01 + 2 * a2
    d2cf2 = 6 * a3 * dcf02 + 2 * a2
    covcf1_2 = s1 * (1 / ((2 * a2) ** 2 - 12 * a1 * a3)) + \
    s2 * (-1 / (3 * a3) + 2 * a2 / (3 * a3 * sqrt((2 * a2) ** 2 - 12 * a1 * a3))) ** 2 + \
    s3 * (a2 / (3 * a3 ** 2) - (1 / (6 * a3 ** 2)) * (6 * a1 * a3 / sqrt((2 * a2) ** 2 - 12 * a1 * a3) + \
    sqrt((2 * a2) ** 2 - 12 * a1 * a3))) ** 2
    covcf2_2 = s1 * (1 / ((2 * a2) ** 2 - 12 * a1 * a3)) + \
    s2 * (-1 / (3 * a3) - 2 * a2 / (3 * a3 * sqrt((2 * a2) ** 2 - 12 * a1 * a3))) ** 2 + \
    s3 * (a2 / (3 * a3 ** 2) + (1 / (6 * a3 ** 2)) * (6 * a1 * a3 / sqrt((2 * a2) ** 2 - 12 * a1 * a3) + \
    sqrt((2 * a2) ** 2 - 12 * a1 * a3))) ** 2

    if d2cf1 < 0:
        cal_shift_cf = dcf01
        cal_stddev_cf = sqrt(covcf1_2)
    else:
        cal_shift_cf = dcf02
        cal_stddev_cf = sqrt(covcf2_2)

    is_all_zero = np.all(np.equal(rsm[1].data[50:90,i,j], 0))
    if is_all_zero == 0:
        return cal_shift_cf * 1.1068
    else:
        return 0
    
#创造一个从红到蓝的颜色条
# 定义颜色段的起始和结束颜色
red = (1, 0, 0, 1)  # 红色，不透明
blue = (0, 0, 1, 1)  # 蓝色，不透明

# 创建颜色列表，这里只定义了两个颜色段：从红色到蓝色
colors = [red, blue]

# 使用分段函数创建colormap
rtb_cmap = mcolors.LinearSegmentedColormap.from_list('red_to_blue', colors)

# 创建一个空矩阵
rows, cols = 2000, 2000  # 例如，10行5列
matrix=np.zeros((rows,cols))


for i in range (1,2000):
    for j in range (1,2000):
        matrix[i,j] = get_dplv(i,j)

normalized_matrix = (matrix - np.mean(matrix)) / (matrix.max() - matrix.min())

# 使用imshow函数绘制图像
plt.imshow(normalized_matrix, cmap=rtb_cmap)
plt.xticks([])
plt.yticks([])

# 显示颜色条
plt.colorbar(ticks=[-1,0,1])

# 显示图像
plt.show()