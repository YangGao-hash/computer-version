import cv2
import numpy as np

class Image_Stitching():
    def __init__(self):
        '''
        @param ratio:  当最近邻距离比值小于 0.85 时，认为该匹配是有效的
        @param min_batch: 设置特征点匹配的最小匹配数量
        @param sift: sift 特征提取器
        @param smoothing_window_size: 设置平滑窗口的大小
        '''
        self.ratio=0.85
        self.min_match=10
        self.sift=cv2.SIFT_create()
        self.smoothing_window_size=800

    def registration(self,img1,img2):
        '''
        提取图像的关键点和描述子，并进行匹配，即找到两幅图像之间的几何关系
        图像可以乱序
        @param img1
        @param img2
        '''
        # 提取关键点和描述符
        kp1, des1 = self.sift.detectAndCompute(img1, None)
        kp2, des2 = self.sift.detectAndCompute(img2, None)

        # 特征点匹配
        matcher = cv2.BFMatcher()
        raw_matches = matcher.knnMatch(des1, des2, k=2)
        
        #筛选较好的匹配点
        good_points = []
        good_matches=[]
        for m1, m2 in raw_matches:
            if m1.distance < self.ratio * m2.distance:
                good_points.append((m1.trainIdx, m1.queryIdx))
                good_matches.append([m1])
        
        # 绘制匹配结果
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
        cv2.imwrite('matching.jpg', img3)
        
        # 如果匹配点数量大于最小匹配数量，则估计单应性变换矩阵
        if len(good_points) > self.min_match:
            image1_kp = np.float32(
                [kp1[i].pt for (_, i) in good_points])
            image2_kp = np.float32(
                [kp2[i].pt for (i, _) in good_points])
            H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
        return H

    def create_mask(self,img1,img2,version):
        '''
        用于在拼接时实现图像的平滑过渡效果
        @param version 区分左右
        '''
        # 获取图像尺寸
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        # 计算平滑窗口大小和偏移量
        offset = int(self.smoothing_window_size / 2)
        barrier = img1.shape[1] - int(self.smoothing_window_size / 2)
        
        # 创建全零蒙版图像
        mask = np.zeros((height_panorama, width_panorama))
        if version== 'left_image':
            # 左侧图像区域
            mask[:, barrier - offset:barrier + offset ] = np.tile(np.linspace(1, 0, 2 * offset ).T, (height_panorama, 1))
            mask[:, :barrier - offset] = 1  # 左侧图像外的区域设置为全白
        else:
            # 右侧图像区域
            mask[:, barrier - offset :barrier + offset ] = np.tile(np.linspace(0, 1, 2 * offset ).T, (height_panorama, 1))
            mask[:, barrier + offset:] = 1  # 右侧图像外的区域设置为全白
        
        return cv2.merge([mask, mask, mask])

    def blending(self,img1,img2):
        '''
            图像融合函数，根据配准结果和蒙版进行图像融合，生成全景图像
        '''
        # 图像配准，获取单应性变换矩阵
        H = self.registration(img1,img2)

        # 获取图像尺寸信息
        height_img1 = img1.shape[0]
        width_img1 = img1.shape[1]
        width_img2 = img2.shape[1]
        height_panorama = height_img1
        width_panorama = width_img1 +width_img2

        # 创建全零全景图像
        panorama1 = np.zeros((height_panorama, width_panorama, 3))
        
        # 创建左侧图像的蒙版
        mask1 = self.create_mask(img1,img2,version='left_image')

        # 将左侧图像拼接到全景图像上，并根据蒙版进行权重调整
        panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
        panorama1 *= mask1
        
        cv2.imwrite('panorama1.jpg', panorama1)
        # 创建右侧图像的蒙版
        mask2 = self.create_mask(img1,img2,version='right_image')
        
        # 对右侧图像进行透视变换，并根据蒙版进行权重调整
        panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
        cv2.imwrite('panorama2.jpg', panorama2)
        
 
        rows, cols=panorama1.shape[:2]
        print(rows)
        print(cols)
        for col in range(0,cols):
            # 开始重叠的最左端
            if panorama1[:, col].any() and panorama2[:, col].any():
                left = col
                print(left)
                break
        # 找出重叠的区域
        for col in range(cols-1, 0, -1):
            #重叠的最右一列
            if panorama1[:, col].any() and panorama2[:, col].any():
                right = col
                print(right)
                break
        
        res = np.zeros([rows, cols, 3], np.uint8)
        for row in range(0, rows):
            for col in range(0, cols):
                if not panorama1[row, col].any():
                    res[row, col] = panorama2[row, col]
                elif not panorama2[row, col].any():
                    res[row, col] = panorama1[row, col]
                else:
                    #if col < img1.shape[1]:
                    left_new = (width_panorama)//2-100
                    right_new = (width_panorama)//2+100
                    if col < left_new:
                        res[row, col] = img1[row, col]
                    elif col > right_new:
                        res[row, col] = img2[row, col]
                    else:
                        # 根据距离计算权重
                        srcImgLen = float(abs(col - left_new))
                        testImgLen = float(abs(col - right_new))
                        alpha = srcImgLen / (srcImgLen + testImgLen)
                        res[row, col] = np.clip(panorama1[row, col] * (1-alpha)*2 + panorama2[row, col] *(alpha)*2, 0, 255)

                     
        return res
        result = res
        # 将两个图像叠加得到最终结果
        # result=panorama1+panorama2

        # return result

        # 剪裁结果，去除全零区域
        rows, cols = np.where(result[:, :, 0] != 0)
        min_row, max_row = min(rows), max(rows) + 1
        min_col, max_col = min(cols), max(cols) + 1
        final_result = result[min_row:max_row, min_col:max_col, :]
        return final_result
    

def main(argv1,argv2):
    img1 = cv2.imread(argv1)
    img2 = cv2.imread(argv2)
    final=Image_Stitching().blending(img1,img2)
    cv2.imwrite('panorama.jpg', final)


if __name__ == '__main__':    
    try: 
        main('img1.jpg', 'img2.jpg')
    except FileNotFoundError as e:
        print("File not found error:", e)
    except Exception as e:
        print("An error occurred:", e)
    
