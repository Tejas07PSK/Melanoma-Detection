import cv2
import numpy as np
import math
from util import Util as u

class Prep(object):

    def __init__(self, path):
        self.__img = cv2.imread(path, cv2.IMREAD_COLOR)
        self.__imgray = cv2.cvtColor(self.__img, cv2.COLOR_BGR2GRAY)
        self.__invimgray = self.__negate()
        self.__ottlvl = self.__OtsuAutoThresh()
        self.__binimg = self.__imBinarize()
        (self.__seg_col, self.__seg_gray) = self.__cvtBinToColAndGray()

    def __negate(self):
        inv_img = (self.__imgray).copy()
        (r, c) = inv_img.shape
        for x in range(0, r, 1):
            for y in range(0, c, 1):
                    inv_img[x,y] = np.invert(inv_img[x,y])
        return inv_img

    def getColorPlates(self, src_clrimg, plate):
            temp_img = src_clrimg.copy()
            for x in temp_img:
                for y in x:
                    if plate == 'B':
                        y[1] = 0
                        y[2] = 0
                    elif plate == 'G':
                        y[0] = 0
                        y[2] = 0
                    elif plate == 'R':
                        y[0] = 0
                        y[1] = 0
            return temp_img

    def __ins(self, arr, ins_val, index, isSearched):
            if (arr.size == 0):
                arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
            else:
                flag = 0
                if (isSearched == 0):
                    fnd_idx = u.search(arr, ins_val, 0, arr.size)
                    if (fnd_idx >= 0):
                        flag = 1
                        ((arr[fnd_idx])[1])[0] = np.uint32(((arr[fnd_idx])[1])[0]) + np.uint32(1)
                if (flag == 1):
                    pass
                elif (ins_val > (arr[index - 1])[0]):
                        arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
                elif (ins_val < (arr[index - 1])[0]):
                        if (index == 0):
                            arr = np.insert(arr, index, (ins_val, np.array([ 1 ], np.uint32)), 0)
                        else:
                            arr = self.__ins(arr, ins_val, index=index - 1, isSearched=1)
                else:
                    ((arr[index - 1])[1])[0] = np.uint32(((arr[index - 1])[1])[0]) + np.uint32(1)
            return arr

    def getArrayOfGrayLevelsWithFreq(self, gray_img):
            aryoflst = np.empty(0, np.dtype([('glvl', np.uint8), ('freq', np.uint32, (1,))]), 'C')
            for x in range(0, (gray_img.shape)[0], 1):
                for y in range(0, (gray_img.shape)[1], 1):
                    aryoflst = self.__ins(aryoflst, gray_img[x, y], index=aryoflst.size, isSearched=0)
            return aryoflst

    def __rmHoles(self, src_binimg):
            ffill_img = src_binimg.copy()
            mask = np.zeros((((ffill_img.shape)[0])+2, ((ffill_img.shape)[1])+2), np.uint8, 'C')
            cv2.floodFill(ffill_img, mask, (0,0), 255);
            final_img = src_binimg | cv2.bitwise_not(ffill_img)
            return final_img

    def __OtsuAutoThresh(self):
        app_grlvls_wth_freq = self.getArrayOfGrayLevelsWithFreq(self.__invimgray)
        dt = np.dtype([('wcv', float), ('bcv', float), ('glvl', np.uint8)])
        var_ary = np.empty(0, dt, 'C')
        for x in range(0, app_grlvls_wth_freq.size, 1):
                  thrslvl = (app_grlvls_wth_freq[x])[0]
                  sumb = 0.0
                  wb = 0.0
                  mb = 0.0
                  varb2 = 0.0
                  sumf = 0.0
                  wf = 0.0
                  mf = 0.0
                  varf2 = 0.0
                  for h in range(x, app_grlvls_wth_freq.size, 1):
                      sumf = sumf + (app_grlvls_wth_freq[h])[1]
                      wf = wf + (app_grlvls_wth_freq[h])[1]
                      mf = mf + float(np.uint64((app_grlvls_wth_freq[h])[0]) * np.uint64((app_grlvls_wth_freq[h])[1]))
                  wf = wf / float((math.pow(app_grlvls_wth_freq.size, 2)))
                  mf = mf / sumf
                  for h in range(x, app_grlvls_wth_freq.size, 1):
                      varf2 = varf2 + float((math.pow((((app_grlvls_wth_freq[h])[0]) - mf), 2)) * ((app_grlvls_wth_freq[h])[1]))
                  varf2 = varf2/sumf
                  if (x == 0):
                      pass
                  else:
                      for h in range(0, x, 1):
                          sumb = sumb + (app_grlvls_wth_freq[h])[1]
                          wb = wb + (app_grlvls_wth_freq[h])[1]
                          mb = mb + float(np.uint64((app_grlvls_wth_freq[h])[0]) * np.uint64((app_grlvls_wth_freq[h])[1]))
                      wb = wb / float((math.pow(app_grlvls_wth_freq.size, 2)))
                      mb = mb / sumb
                      for h in range(0, x, 1):
                          varb2 = varb2 + float((math.pow((((app_grlvls_wth_freq[h])[0]) - mb), 2)) * ((app_grlvls_wth_freq[h])[1]))
                      varb2 = varb2 / sumb
                  wcv = (wb * varb2) + (wf * varf2)
                  bcv = (wb * wf) * math.pow((mb - mf), 2)
                  var_ary = np.append(var_ary, np.array([(wcv, bcv, thrslvl)], dtype=dt), 0)
        u.quickSort(var_ary, 0, var_ary.size - 1)
        ottlvl = (var_ary[0])[2]
        return ottlvl

    def __imBinarize(self):
        binimg = np.zeros((self.__invimgray).shape, np.uint8, 'C')
        for x in range(0, ((self.__invimgray).shape)[0], 1):
            for y in range(0, ((self.__invimgray).shape)[1], 1):
                if (self.__invimgray[x, y] < self.__ottlvl):
                    binimg[x, y] = np.uint8(0)
                else:
                    binimg[x, y] = np.uint8(255)
        binimg = self.__rmHoles(binimg)
        return binimg

    def __cvtBinToColAndGray(self):
        seg_col = np.zeros((self.__img).shape, np.uint8, 'C')
        seg_gray = np.zeros((self.__imgray).shape, np.uint8, 'C')
        i = 0
        for x in seg_col:
            j = 0
            for y in x:
                if ((self.__binimg)[i, j] == 255):
                    y[0] = (self.__img)[i, j, 0]
                    y[1] = (self.__img)[i, j, 1]
                    y[2] = (self.__img)[i, j, 2]
                    seg_gray[i, j] = self.__imgray[i, j]
                j = j + 1
            i = i + 1
        return (seg_col, seg_gray)

    def getActImg(self):
        return self.__img

    def getGrayImg(self):
        print ((self.__imgray).size)
        return self.__imgray

    def getInvrtGrayImg(self):
        print ((self.__invimgray).size)
        return self.__invimgray

    def getBinaryImg(self):
        print ((self.__binimg).size)
        return self.__binimg

    def getOtsuThresholdLevel(self):
        return self.__ottlvl

    def getSegColImg(self):
        print ((self.__seg_col).size)
        return self.__seg_col

    def getSegGrayImg(self):
        print ((self.__seg_gray).size)
        return self.__seg_gray
