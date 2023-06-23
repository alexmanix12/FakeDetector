import cv2
import numpy as np

LOWER_LIMIT = 0.2
UPPER_LIMIT = 0.6


class BackDetector:
    def __init__(self):
        pass 

    def find_foreground(self, image_path):

        # Read Image and its edges using Canny edge detection
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canned = cv2.Canny(gray, 100, 200)
        kernel = np.ones((5,5),np.uint8)
        mask = cv2.dilate(canned, kernel, iterations = 1)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Finding the detailed area

        crop_mask = np.zeros_like(mask)

        biggest_cntr = None
        biggest_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > biggest_area:
                biggest_area = area
                biggest_cntr = contour

        cv2.drawContours(crop_mask, [biggest_cntr], -1, (255), -1)

        # Filling gaps in detailed area

        inverted = cv2.bitwise_not(crop_mask)
        contours, _ = cv2.findContours(inverted, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        small_cntrs = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 20000:
                small_cntrs.append(contour)

        cv2.drawContours(crop_mask, small_cntrs, -1, (255), -1)

        return crop_mask

    def is_blurred(self, image_path):
        foreground = self.find_foreground(image_path)
        perc = np.count_nonzero(foreground == 0) / foreground.size
        return perc >= 0.2 and perc <= 0.6

    def testFolder(self, folder_path):
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        return zip(images, [is_blurred(image) for image in images])