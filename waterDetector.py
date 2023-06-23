import cv2
import os
from os import path
from imwatermark import WatermarkDecoder
from PIL import Image
from PIL.ExifTags import TAGS
from transformers import pipeline

class WaterDetector:
    def __init__(self):
        self.decoder = WatermarkDecoder('bytes', 136)
        self.pipe = pipeline("image-classification", "umm-maybe/AI-image-detector")

    # Returns 1 if evidence of the picture being genereated is found, 0 if no
    # evidence and -1 if contrary evidence
    def testImage(self, image_path):
        meta = self.__testMeta(image_path)
        if meta == 1:
            return self.__testInvis(image_path) or self.__testVis(image_path)
        return meta

    # Returns 1 if the image contains an invisible watermark
    def __testInvis(self, image_path):
        bgr = cv2.imread(image_path)
        watermark = self.decoder.decode(bgr, "dwtDct")
        try:
            dec = watermark.decode('utf-8')
            return 1
        except:
            return 0

    # Return True if the image contains a visible watermark
    def __testVis(self, image_path):
        img = cv2.imread(image_path)
        outputs = self.pipe(img)
        results = {}
        for result in outputs:
            results[result['label']] = result['score']

        return results['artificial'] >= 0.95

    # Returns 1 if the image does not contain tags for the model of camera used
    # or the software used to make/edit the picture, -1 otherwise
    def __testMeta(self, image_path):
        image = Image.open(image_path)
        exifdata = image.getexif()

        tags = [TAGS.get(tag_id, tag_id) for tag_id in exifdata]
        real_tags = ["Model", "Software"]

        for tag in real_tags:
            if tag in tags:
                return -1

        return 1

    def testFolder(self, folder_path):
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        results = [self.testImage(img_path) for img_path in images]
        return zip(images, results)