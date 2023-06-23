import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
from gan_detect_iris.crop_eyes import crop_eye, drawPoints, eye_detection
from gan_detect_iris.crop_cornea import cornea_convex_hull
from gan_detect_iris.crop_iris import  segment_iris
from gan_detect_iris.crop_highlights import process_aligned_image

def Detection(args):
    #### 1. Read image
    try:
        data_name = os.path.splitext(os.path.basename(args['input']))[0]
        # data_name = args.input.split("/")[-1]
        # data_name = data_name.split(".")[0]
    except:
        print ('The input image path or name is not correct. Please rename your image as name.type.')
        return False
        # exit()
    #### 2. Crop eye
    try:
        left_eye_image, right_eye_image, new_eyes_position_list, number_face, double_eye_img, double_eye_position_difference_list \
            = eye_detection(args['input'], args['predictor_path'])
    except:
        print ('Your image has some problems. It may not contain faces.')
        return False
        # exit()
    if number_face != 1:
        print ('Your image contains more than one face. However, our software can only work on one face.')
        return False
        # exit()
    #### 3. Crop cornea
    try:
        left_cornea, right_cornea, left_cornea_matrix, right_cornea_matrix \
            = cornea_convex_hull(left_eye_image, right_eye_image, new_eyes_position_list)
    except:
        logger.error('Crop cornea failed.')
        return False
        # exit()
    #### 4. Crop iris
    try:
        img_left, iris_left, l_iris, l_iris_center, l_radius, l_eye_center, l_highlights, l_num_refl, l_valid \
            = segment_iris(left_eye_image, left_cornea_matrix.astype(bool), args['radius_min_para'],
                           args['radius_max_para'])  # 'left'
        img_right, iris_right, r_iris, r_iris_center, r_radius, r_eye_center, r_highlights, r_num_refl, r_valid \
            = segment_iris(right_eye_image, right_cornea_matrix.astype(bool), args['radius_min_para'],
                           args['radius_max_para'])  # 'right'

        if l_num_refl==0 and r_num_refl==0:
            return False
    except:
        print ('Crop iris failed.')
        return False

        # exit()
    #### 5. Draw circles on double eyes
    # try:
    #     double_eye_img_ori = double_eye_img.copy()
    #     new_left_eye = l_iris_center + double_eye_position_difference_list[0]
    #     new_right_eye = r_iris_center + double_eye_position_difference_list[1]
    #     cv2.circle(double_eye_img, (new_left_eye[0], new_left_eye[1]), l_radius, (0, 0, 255), 2)  # left
    #     cv2.circle(double_eye_img, (new_right_eye[0], new_right_eye[1]), r_radius, (0, 0, 255), 2)  # right
    # except:
    #     print ('Draw circles on double eyes failed.')
    #     return False
        # exit()
    #### 6. Crop highlights
    try:
        iris_left_resize, iris_right_resize, left_recolor, right_recolor, \
        left_recolor_resize, right_recolor_resize, IOU_score, double_eye_img_modified \
            = process_aligned_image(iris_left, iris_right, l_iris, r_iris, l_highlights, r_highlights, left_eye_image,
                                    right_eye_image,
                                    double_eye_img, double_eye_position_difference_list, reduce=args['shrink'],
                                    reduce_size=args['shrink_size'], threshold_scale_left=args['threshold_scale_left'],
                                    threshold_scale_right=args['threshold_scale_right'])
    except:
        print ('Crop highlights failed.')
        return False
        # exit()

    # #### 7. Save result
    # try:
    #     ori_image = cv2.imread(args.input)
    #     ori_image = cv2.resize(ori_image, (double_eye_img_ori.shape[1], double_eye_img_ori.shape[1]))
    #     ori_image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    #     space = np.full((2, double_eye_img_ori.shape[1], 3), 255, dtype=np.uint8)
    #     imgs_comb = np.vstack((ori_image, space, double_eye_img_ori, space, double_eye_img_modified))
    #     imgs_comb = Image.fromarray(imgs_comb)
    #     plt.imshow(imgs_comb)
    #     plt.xticks([])
    #     plt.yticks([])
    #     plt.xlabel("IoU:{}".format(f'{IOU_score:.4f}'))
    #     os.makedirs(args.output, exist_ok=True)
    #     plt.savefig('{}/{}_iris_final.png'.format(args.output, data_name), dpi=800, bbox_inches='tight',
    #                 pad_inches=0)
    #     plt.show()
    #     logger.info("IOU:{}".format(f'{IOU_score:.4f}'))
    #     logger.info("The result is saved in {}/{}_iris_final.png".format(args.output, data_name))
    # except:
    #     print ('Save result failed.')
    #     return False

    return IOU_score


def getIOU(image_path):
    args = {
        'input' : image_path,
        'radius_min_para' : 4.5,
        'radius_max_para' : 2.0,
        'shrink' : True,
        'shrink_size' : 2,
        'threshold_scale_left' : 1.2,
        'threshold_scale_right' : 1.2,
        'predictor_path' :  "shape_predictor/shape_predictor_68_face_landmarks.dat"
        }
    IOU_score = Detection(args)
    return IOU_score

class EyeDetector:
    def testImage(self, image_path):
        return getIOU(image_path) < 0.5

    def testFolder(self, folder_path):
        images = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
        return zip(images, [self.testImage(img) for img in images])