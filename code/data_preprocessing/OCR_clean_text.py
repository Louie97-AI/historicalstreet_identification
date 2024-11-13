import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import cv2
import rasterio
from paddleocr import PaddleOCR, draw_ocr
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from tqdm import tqdm


class Clean_text(object):
    
    def __init__(self,path):
        huge_img = cv2.imread(path)
        height,width = huge_img.shape[:2]
        
        self.height = height
        self.width = width
        self.img = huge_img
        self.ocr = PaddleOCR()

    def binary_map(self):
        gray_huge_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        _, self.img = cv2.threshold(gray_huge_image, 150, 255, cv2.THRESH_BINARY)

    
    def after_cleaning(self,small_photo_size,start_coords,save_S_folder_path,save_L_folder_path,index):
        img = self.img
        start__x = start_coords[0]
        start__y = start_coords[1]
        
        count = 0
        for y in tqdm(range(start__y, self.height, small_photo_size)):
            for x in range(start__x, self.width, small_photo_size):
                small_img = img[y:y + small_photo_size, x:x + small_photo_size]
                
                tile_path = r"{}\small_{}_{}.jpg".format(save_S_folder_path,y,x)  
                              
                cv2.imwrite(tile_path, small_img)

                result = self.ocr.ocr(tile_path, cls=False)
                
                if len(result)>0 and None not in result:
                    
                    for item in result[0]:
                        rect_coords = item[0]
                        rect_coords = [[int(jj[0]),int(jj[1])] for jj in rect_coords]
                        
                        mask = np.zeros_like(small_img)

                        cv2.fillPoly(mask, [np.array(rect_coords)], (255, 255, 255))

                        maskimg_result = cv2.addWeighted(small_img, 1, mask, 1, 0)
                        
                        small_img = maskimg_result
                        
                        cv2.imwrite(tile_path, maskimg_result)
                        
                        count+=1


#---------------------------------Image Collage: Restoring Small Images into a Large Image-------------------------------------------------------------#
        reconstructed_image = np.zeros((self.height, self.width), dtype=np.uint8)

        for y in range(start__y, self.height, small_photo_size):
            for x in range(start__x, self.width, small_photo_size):
                tile_path = r"{}\small_{}_{}.jpg".format(save_S_folder_path,y,x)  
                
                small_img = cv2.imread(tile_path, cv2.IMREAD_GRAYSCALE)
                reconstructed_image[y:y + small_photo_size, x:x + small_photo_size] = small_img

        path_concat = r"{}\reconstructed_image_{}_{}.jpg".format(save_L_folder_path,index,count)
        cv2.imwrite(path_concat, reconstructed_image)
        
        return path_concat,count



if __name__ =="__main__":

# Repeatedly Loop to Test Different Combinations

    base_folder = r"E:\urbanquantityresearch\recover_yuan_road\test_data\iteration_folder"
    huge_map_path_list = [r"E:\urbanquantityresearch\recover_yuan_road\test_data\iteration_folder\save_L_map\hugemap.png"]
   
    # parameter_list
    parameter_list = [[100, [0,0]],[100, [50,50]],[100, [100,100]],[100, [150,150]],
                      [150, [50,50]],[150, [100,100]],[150, [150,150]],[150, [200,200]],
                      [200, [0,0]],[200, [50,50]],[200, [100,100]],[200, [150,150]],
                      [50, [25,25]],[50, [100,100]],[50, [150,150]],[50, [200,200]],
                      [25,[0,0]],[25,[50,50]],[25,[100,100]],[25,[200,200]],
                      [300, [50,50]],[300, [100,100]],[300, [150,150]]
                      ]
    
    save_L_folder_path = r"E:\urbanquantityresearch\recover_yuan_road\test_data\iteration_folder\save_L_map"

    result_list = {}

    for index,item in enumerate(parameter_list):
        
        # make new save small img dir
        save_S_folder_path = base_folder+f"\save_S_map_{index}"
        os.mkdir(save_S_folder_path)
        
        huge_map_path = huge_map_path_list[0]
        huge_map_path_list.pop(0)
        
        c_map = Clean_text(huge_map_path)
        new_hugeimg_p,counts = c_map.after_cleaning(item[0], item[1], save_S_folder_path, save_L_folder_path,index)
        
        huge_map_path_list.append(new_hugeimg_p)
        
        result_list[f"{item}"] = counts












print