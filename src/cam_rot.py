import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

# def SubtractionBg():
#     cap = cv2.VideoCapture("frame%08d.png", cv2.CAP_IMAGES)
#     history = 20
#     fgbg = cv2.createBackgroundSubtractorKNN(detectShadows=True)
#     fgbg.setHistory(history)
#     frames=0
#     for i in range(0,50):
#         ret, frame = cap.read()
#         fgmask = fgbg.apply(frame)
#         if frames < history:
#             frames += 1
#             continue
#         th = cv2.threshold(fgmask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
#         th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
#         dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)        # 获取所有检测框
#         image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)        
#         for c in contours:            # 获取矩形框边界坐标
#             x, y, w, h = cv2.boundingRect(c)            # 计算矩形框的面积
#             area = cv2.contourArea(c)            
#             if 500 < area < 3000:
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         cv2.imshow("detection", frame)
#         cv2.imshow("back", dilated)
#         if i==50:
#             break
#         #cv2.imshow('background', background)
#         k = cv2.waitKey(10)&0xff
#         if k == 27:
#             break
#     cap.release()
#     #cv2.destroyAllWindows()
def cameraPoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3])


MIN_MATCH_COUNT = 10
totalFrame = 59 # totalFrame
CamRotAngles = [] # result

def PredictTheta(fname1, fname2):
    img1 = cv2.imread(fname1,0)
    img2 = cv2.imread(fname2,0)

    orb = cv2.ORB_create()
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        # print(src_pts.shape)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        #print(M)
        # print(np.linalg.det(M))

        EigenValue, EigenVector= np.linalg.eig(M)
        EigenValue = EigenValue.tolist()
        print('--------------------------------------')
        print(EigenValue)
        # print(EigenVector)
        # print('--------------------------------------')
        RealValue = 0
        times = 0
        for index,i in enumerate(EigenValue):
            if(math.isclose(i.imag,0.0)):
                del EigenValue[index]
                RealValue = i.real
                times+=1
        if(times!=1):
            print("Not right!!!!" + str(times)) # it should only be exactly 1 real eigen value
            CamRotAngles.append(0)
            return
        # print("RealValue is : " + '%.6f' %RealValue)
        # print("After Remove Real EigenValues are :" + str(EigenValue))
        first = EigenValue[0]
        second = EigenValue[1]
        first = np.complex(first.real/RealValue, first.imag/RealValue)
        # print("Square sum is : " + str(np.abs(first)))
        Theta = math.atan2(first.imag, first.real)
        print(fname1 + " Theta is : " + str(Theta))
        
        if(Theta>1.0): # 向右转
            Theta = -Theta
        CamRotAngles.append(Theta)

        matchesMask = mask.ravel().tolist()

        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)

        img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    else:
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)

    img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

    ''' if to show the matching point '''

    # plt.imshow(img3, 'gray'),plt.show()


if __name__ == "__main__":
    frameindex = 0 # starts from 0    
    for i in range(0, totalFrame):
        fname1 = 'frame'+str(frameindex).zfill(8)+'.png'
        fname2 = 'frame' + str(frameindex+1).zfill(8)+ '.png'
        PredictTheta(fname1,fname2)
        frameindex+=1
    if len(CamRotAngles)!= totalFrame :
        print(len(CamRotAngles))
        print("Something Wrong!!!!!!!!!")
    else:
        with open('camrot.json','w') as f:
            import json
            json.dump(CamRotAngles,f)
            print("Done!")
