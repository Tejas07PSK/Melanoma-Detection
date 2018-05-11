import cv2
import numpy as np
import math
from util import Util as u

class Prep(object):
        __imcrop = np.zeros(1,np.uint8,'C')

        def __init__(self,path):
            self.__img = cv2.imread(path,cv2.IMREAD_COLOR)
            self.__imgray = cv2.cvtColor(self.__img,cv2.COLOR_BGR2GRAY)
            self.__invimgray = self.ivnert(self.__imgray)
            self.__binimg = np.zeros((self.__invimgray).shape,np.uint8,'C')

        def ivnert(self,src_gimg):
            inv_img = src_gimg.copy()
            (r,c) = inv_img.shape
            for x in range(0,r,1):
                for y in range(0,c,1):
                    inv_img[x,y] = np.invert(inv_img[x,y])
            return inv_img

        def getColorPlates(self,src_clrimg,plate):
            temp_img = src_clrimg.view()
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

        def ins(self,arr, ins_val, index, isSearched):
            if (arr.size == 0):
                arr = np.insert(arr, index, (ins_val, 1), 0)
            else:
                if (isSearched == 0):
                    fnd_idx = u.search(arr, ins_val, 0, arr.size)
                    if (fnd_idx >= 0):
                        arr[fnd_idx] = (((arr[fnd_idx])[0]), ((arr[fnd_idx])[1] + 1))
                if (ins_val > (arr[index - 1])[0]):
                    arr = np.insert(arr, index, (ins_val, 1), 0)
                elif (ins_val < (arr[index - 1])[0]):
                    if (index == 0):
                        arr = np.insert(arr, index, (ins_val, 1), 0)
                    else:
                        arr = self.ins(arr, ins_val, index=index - 1, isSearched=1)
            return (arr)

        def getArrayOfGrayLevelsWithFreq(self,gray_img):
            aryoflst = np.empty(0, np.dtype([('glvl', np.uint8), ('freq', np.uint8)]), 'C')
            for x in range(0,(gray_img.shape)[0],1):
                for y in range(0,(gray_img.shape)[1],1):
                    aryoflst = self.ins(aryoflst,gray_img[x,y],index=aryoflst.size,isSearched=0)
            return (aryoflst)

        def rmHoles(self,src_binimg):
            ffill_img = src_binimg.copy()
            mask = np.zeros((((ffill_img.shape)[0])+2, ((ffill_img.shape)[1])+2), np.uint8, 'C')
            cv2.floodFill(ffill_img, mask, (0,0), 255);
            final_img = src_binimg | cv2.bitwise_not(ffill_img)
            return (final_img)

        def OtsuAutoThresh(self,src_invgrimg):
            print(src_invgrimg)
            app_grlvls_wth_freq = self.getArrayOfGrayLevelsWithFreq(src_invgrimg)
            dt=np.dtype([('wcv',float),('bcv',float),('glvl',np.uint8)])
            var_ary = np.empty(0,dt,'C')
            print(app_grlvls_wth_freq)
            for x in range(0,app_grlvls_wth_freq.size,1):
                  thrslvl = (app_grlvls_wth_freq[x])[0]
                  sumb = 0.0
                  wb = 0.0
                  mb = 0.0
                  varb2 = 0.0
                  sumf = 0.0
                  wf = 0.0
                  mf = 0.0
                  varf2 = 0.0
                  for h in range(x,app_grlvls_wth_freq.size,1):
                      sumf = sumf + (app_grlvls_wth_freq[h])[1]
                      wf = wf + (app_grlvls_wth_freq[h])[1]
                      mf = mf + float(np.uint64((app_grlvls_wth_freq[h])[0]) * np.uint64((app_grlvls_wth_freq[h])[1]))
                  wf = wf / float((math.pow(app_grlvls_wth_freq.size,2)))
                  mf = mf / sumf
                  for h in range(x,app_grlvls_wth_freq.size,1):
                      varf2 = varf2 + float((math.pow((((app_grlvls_wth_freq[h])[0]) - mf),2)) * ((app_grlvls_wth_freq[h])[1]))
                  varf2 = varf2/sumf
                  if (x == 0):
                      pass
                  else:
                      for h in range(0,x,1):
                          sumb = sumb + (app_grlvls_wth_freq[h])[1]
                          wb = wb + (app_grlvls_wth_freq[h])[1]
                          mb = mb + float(np.uint64((app_grlvls_wth_freq[h])[0]) * np.uint64((app_grlvls_wth_freq[h])[1]))
                      wb = wb / float((math.pow(app_grlvls_wth_freq.size, 2)))
                      mb = mb / sumb
                      for h in range(0,x,1):
                          varb2 = varb2 + float((math.pow((((app_grlvls_wth_freq[h])[0]) - mb), 2)) * ((app_grlvls_wth_freq[h])[1]))
                      varb2 = varb2 / sumb
                  wcv = (wb * varb2) + (wf * varf2)
                  bcv = (wb * wf) * math.pow((mb - mf),2)
                  var_ary = np.append(var_ary,np.array([(wcv,bcv,thrslvl)],dtype=dt),0)
            print(var_ary)
            u.quickSort(var_ary,0,var_ary.size - 1)
            print(var_ary)
            return (var_ary,app_grlvls_wth_freq)

        def imBinarize(self,src_invgrimg):
            arset = self.OtsuAutoThresh(src_invgrimg)
            for x in range(0,(src_invgrimg.shape)[0],1):
                for y in range(0,(src_invgrimg.shape)[1],1):
                     if (src_invgrimg[x,y] < ((arset[0])[0])[2]):
                         self.__binimg[x,y] = np.uint8(0)
                     else:
                         self.__binimg[x,y] = np.uint8(255)
            print(self.__binimg)
            self.__binimg = self.rmHoles(self.__binimg)
            return (arset[0],arset[1])

        def getActImg(self):
            return self.__img

        def getGrayImg(self):
            return self.__imgray

        def getInvrtGrayImg(self):
            return self.__invimgray

        def getBinaryImg(self):
            return self.__binimg

