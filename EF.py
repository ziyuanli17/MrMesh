# -*- coding: utf-8 -*-
"""DeMRI_Processing Test

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1cg2Ey3Baw0ftmwEFe9GoxuVkWy8M-2mt
"""
import numpy as np
import cv2 as cv
import os
import nibabel as nib
from PIL import Image, ImageFilter
import json
import string
from contour_extraction import extract_contour
from PIL import Image

print("Copyright (c) 2022 Silva Lab")
print("All Rights Reserved")
# Load parameters
parameter_dict = json.loads(open('parameters.json', 'r').read())
"""# **Software Parameters**"""

# Required parameters to modify
Loc_idx = parameter_dict["Loc_idx"][
    0]  # The n-th slice FCN-Net generated segmentation (required if using FCN localization method)

# Recommended parameters to modify (In case performance is not satisfactory)
erode_iterations = parameter_dict["erode_iterations"][
    0]  # Number of iterations to erode (A larger value leads to more degree of erosion)
color1 = parameter_dict["color1"][0]  # Lower color limit for color filtering
blockSize = parameter_dict["blockSize"][0]  # Determines the size of the neighbourhood area

# Optional parameters to modify (In case performance is not satisfactory)
color2 = parameter_dict["color2"][0]  # Upper color limit for color filtering
C = parameter_dict["C"][
    0]  # A constant that is subtracted from the mean or weighted sum of the neighbourhood pixels
area1 = parameter_dict["area1"][0]  # Lower area limit for black area filtering
area2 = parameter_dict["area2"][0]  # Lower area limit for white area filtering
area3 = parameter_dict["area3"][0]  # Upper limit for white area filtering
LV_dilate_add = parameter_dict["LV_dilate_add"][0]  # To artificially increase the LV size
Myo_dilate_add = parameter_dict["Myo_dilate_add"][0]  # To artificially increase the Myo size
hough_thresh = 20  # Threshold for Hough circle detection (larger would increase the chance for a circular shape to be detected)
minR = 5  # Lower radius limit for circle detection
maxR = 18  # Upper radius limit for circle detection
outlier_threshold = parameter_dict["outlier_thresh"][0]  # Smaller value leads to more outliers being detected

start_idx = Loc_idx
end_idx = Loc_idx

"""# Load images and masks"""
slice_locs = parameter_dict["slice_locs"]


# Utils function to load and save nifti files with the nibabel package
def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_fdata(), nimg.affine, nimg.header


# for i in range(len(imgs_SA[0][0])):
#   img = imgs_SA[:, :, i]

#   np.clip(img, 0, 255, out=img)
#   img = img.astype('uint8')
#   img = cv.equalizeHist(img)
#   cv2_imshow(img)
#   imgs_SA[:, :, i] = img


# masks_LA, _, _ = load_nii(path+"seg_la_4ch_ED.nii.gz")
# imgs_LA, _, _ = load_nii(path+"la_4ch_ED.nii.gz")

"""# **LV and Myo Segmentation Functions**"""


# Remove colors of selection
def bandpass_filter(img, color1=225, color2=800, reverse=False):
    # Replace out-off-range colors with black
    img[img < 30] = 0
    if reverse:
        img[img >= color1] = 0
    else:
        img[img < color1] = 0
        img[img > color2] = 0
    return img


# Take input image and area, remove any objects smaller than the defined area
def undesired_objects(binary_map, area, check):
    # do connected components processing
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(cv.bitwise_not(binary_map), None, None, None, 8,
                                                                        cv.CV_32S)

    # get CC_STAT_AREA component as stats[label, COLUMN]
    areas = stats[1:, cv.CC_STAT_AREA]

    result = np.zeros((labels.shape), np.uint8)
    for i in range(0, nlabels - 1):
        if areas[i] >= area:  # keep
            result[labels == i + 1] = 255
        if areas[i] >= area3 and check == 2:
            result[labels == i + 1] = 0
    return cv.bitwise_not(result)


# Filter out both black and white connected components
def filter_2D(img, area1=1000, area2=100):
    # Creating kernel
    kernel = np.ones((2, 2), np.uint8)
    th2 = cv.erode(img, kernel, iterations=1)
    # Remove small objects two-directionally
    result = undesired_objects(th2, area1, 1)
    # print("Black components filtered")
    # cv2_imshow(result)
    result = cv.bitwise_not(undesired_objects(cv.bitwise_not(result), area2, 2))
    # print("White components filtered")
    # cv2_imshow(result)
    result = cv.dilate(result, kernel, iterations=1)
    return result


# Hough localization to detect LA center
def hough_localization(img_series):
    c_x = []
    c_y = []
    for img in img_series:
        # Hough localization
        img_blur = cv.medianBlur(img.copy(), 5)
        img_blur = (img_blur / np.max(img_blur)) * 255
        np.clip(img_blur, 0, 255, out=img_blur)
        img_blur = img_blur.astype('uint8')
        rows = img_blur.shape[0]
        circles = cv.HoughCircles(img_blur, cv.HOUGH_GRADIENT, 1, rows, param1=80, param2=hough_thresh, minRadius=minR,
                                  maxRadius=maxR)
        RA_center = None
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for c in circles[0, :]:
                # draw the outer circle
                cv.circle(img_blur, (c[0], c[1]), c[2], (0, 255, 0), 2)
                # draw the center of the circle
                cv.circle(img_blur, (c[0], c[1]), 2, (0, 0, 255), 3)
                c_x.append(c[0])
                c_y.append(c[1])
            # cv2_imshow(img_blur)
        RA_center = [np.median(c_x), np.median(c_y)]
        return RA_center


# Find closest point in array
def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)


# Find nearest value in array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# Identify and out RA and LA mask
def chamber_localization(img, LA_center):
    nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img, None, None, None, 4, cv.CV_32S)
    # If not enough regions can be identified
    if LA_center is None:
        return None, None, None, None, None

    centroids = centroids[1:]
    if nlabels == 6:
        # Sort centroids and identify the RA and LA
        draw_image = img.copy()
        centroids = centroids[centroids[:, 1].argsort()]
        c_RA = centroids[3]
        RA_label = labels[int(c_RA[1]), int(c_RA[0])]
        cv.circle(draw_image, (int(c_RA[0]), int(c_RA[1])), 1, (0, 0, 0), 1)
        # print("Localized ")
        # cv2_imshow(draw_image)

        centroids_top = centroids[0:3]
        c_LA = centroids_top[centroids_top[:, 0].argsort()][-1]
        LA_label = labels[int(c_LA[1]), int(c_LA[0])]
        cv.circle(draw_image, (int(c_LA[0]), int(c_LA[1])), 1, (0, 0, 0), 1)
        # cv2_imshow(draw_image)

    else:
        draw_image = img.copy()
        c_LA = centroids[closest_node(LA_center, centroids)]

        LA_label = labels[int(c_LA[1]), int(c_LA[0])]
        cv.circle(draw_image, (int(c_LA[0]), int(c_LA[1])), 1, (0, 0, 0), 1)
        # print("Localized ")
        # cv2_imshow(draw_image)

        centroids_y = centroids[:, 1]
        below_LA_pts = centroids[(centroids_y > c_LA[1]) & (centroids_y > (c_LA[1] + 25))]
        below_LA_x = below_LA_pts[:, 0]
        below_LA_pts = below_LA_pts[(below_LA_x > (c_LA[0] + 30)) | (below_LA_x > (c_LA[0] - 30))]
        try:
            c_RA = below_LA_pts[below_LA_pts[:, 1].argsort()][0]
            RA_label = labels[int(c_RA[1]), int(c_RA[0])]
            cv.circle(draw_image, (int(c_RA[0]), int(c_RA[1])), 1, (0, 0, 0), 1)
            # cv2_imshow(draw_image)
        except:
            return None, None, None, None, None
    # get CC_STAT_AREA component as stats[label, COLUMN] 
    areas = stats[1:, cv.CC_STAT_AREA]
    result = np.zeros((labels.shape), np.uint8)
    result[labels == RA_label] = 255
    result[labels == LA_label] = 75
    # cv2_imshow(result)
    dilated_result = cv.dilate(result, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
                               iterations=erode_iterations + 1)
    # cv2_imshow(dilated_result)

    return areas[LA_label - 1], areas[RA_label - 1], c_LA, c_RA, dilated_result


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def reject_outliers(data, m):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else 0.
    return (s < m)


def save_gif(ds_sa_ed_img, output_folder):
    ds_sa_ed_img = np.array(ds_sa_ed_img)
    for i in range(np.shape(ds_sa_ed_img)[0]):
        img = ds_sa_ed_img[i, :, :].astype('uint16')
        if np.max(img) != 0:
            img = ((img / np.max(img)) * 255).astype('uint16')
        im = Image.fromarray(img)
        im.save(output_folder + str(string.ascii_lowercase[i]) + ".gif")


def save_dicom(img_arrays, output_folder, output_name):
    img_arrays = np.array(img_arrays).astype('uint16')
    img_arrays = np.transpose(img_arrays, (1, 2, 0))
    new_image = nib.Nifti1Image(img_arrays, affine=np.eye(4))
    nib.save(new_image, output_folder + output_name)


def delete_all_files(path):
    for f in os.listdir(path):
        os.remove(os.path.join(path, f))


def save_img(img, name):
    try:
        os.mkdir("ef_process")
    except FileExistsError:
        pass

    im = Image.fromarray(img)
    im.save("ef_process/" + name + ".gif")


def format_img(img, to_gray=True):
    np.clip(img, 0, 255, out=img)
    img = img.astype('uint8')
    if to_gray:
        try:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        except:
            pass
    else:
        try:
            img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        except:
            pass
    return img


def edge_detection(img, seg):
    img = format_img(img, False)
    seg = format_img(seg)

    seg_myo = seg.copy()
    seg_myo[seg_myo != 0] = 255
    seg_lv = seg.copy()
    seg_lv[seg_lv == 255] = 0
    seg_lv[seg_lv != 0] = 255

    i = 0
    for s in [seg_myo, seg_lv]:
        # Blur the image for better edge detection
        img_blur = cv.GaussianBlur(s, (5, 5), 0)
        # Canny Edge Detection
        edges = cv.Canny(image=img_blur, threshold1=0, threshold2=0)  # Canny Edge Detection

        contours, h = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

        big_contour = max(contours, key=cv.contourArea)
        M = cv.moments(big_contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        cv.drawContours(img, contours, -1, (0, 0 + i, 255 - i), thickness=1, lineType=cv.LINE_AA)
        i += 255

    # Display Canny Edge Detection Image
    return img


"""# **LV and Myo Segmentation**"""


def run_ef_segmentation():
    try:
        delete_all_files("ef_process")
    except:
        pass
    # Load parameters
    parameter_dict = json.loads(open('parameters.json', 'r').read())
    """# **Software Parameters**"""

    # Required parameters to modify
    Loc_idx = parameter_dict["Loc_idx"][
        0]  # The n-th slice FCN-Net generated segmentation (required if using FCN localization method)

    # Recommended parameters to modify (In case performance is not satisfactory)
    erode_iterations = parameter_dict["erode_iterations"][
        0]  # Number of iterations to erode (A larger value leads to more degree of erosion)
    color1 = parameter_dict["color1"][0]  # Lower color limit for color filtering
    blockSize = parameter_dict["blockSize"][0]  # Determines the size of the neighbourhood area

    # Optional parameters to modify (In case performance is not satisfactory)
    color2 = parameter_dict["color2"][0]  # Upper color limit for color filtering
    C = parameter_dict["C"][
        0]  # A constant that is subtracted from the mean or weighted sum of the neighbourhood pixels
    area1 = parameter_dict["area1"][0]  # Lower area limit for black area filtering
    area2 = parameter_dict["area2"][0]  # Lower area limit for white area filtering
    area3 = parameter_dict["area3"][0]  # Upper limit for white area filtering
    LV_dilate_add = parameter_dict["LV_dilate_add"][0]  # To artificially increase the LV size
    Myo_dilate_add = parameter_dict["Myo_dilate_add"][0]  # To artificially increase the Myo size
    hough_thresh = 20  # Threshold for Hough circle detection (larger would increase the chance for a circular shape to be detected)
    minR = 5  # Lower radius limit for circle detection
    maxR = 18  # Upper radius limit for circle detection
    outlier_threshold = parameter_dict["outlier_thresh"][0]  # Smaller value leads to more outliers being detected

    start_idx = Loc_idx
    end_idx = Loc_idx

    path = "Output/" + parameter_dict["name"].split(".")[0] + "/"
    masks_SA, _, header = load_nii(path + "seg_sa_ED.nii.gz")
    imgs_SA, _, header = load_nii(path + "sa_ED.nii.gz")

    idx_array = np.array(list(range(start_idx - 1, end_idx)))
    SA_mask_ED = []
    SA_img_ED = []
    SA_LV_mask_ED = []
    slice_locs_trimed = []
    MYO_areas = []
    MYO_centers = []

    # Find nn mask center
    if parameter_dict["lv_loc"] == []:
        nn_mask_LV = masks_SA[:, :, Loc_idx].copy()
        nn_mask_LV[nn_mask_LV == 2] = 0  # Myo
        nn_mask_LV[nn_mask_LV == 3] = 0  # RV
        nn_mask_LV[nn_mask_LV == 1] = 255  # LV
        np.clip(nn_mask_LV, 0, 255, out=nn_mask_LV)
        nn_mask_LV = nn_mask_LV.astype('uint8')
        # cv2_imshow(nn_mask_LV)
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(nn_mask_LV, None, None, None, 4, cv.CV_32S)

        area_nn = 1

        c_LV = centroids[-1]
    else:
        c_LV = parameter_dict["lv_loc"]
        area_nn = 1

    for i in range(masks_SA.shape[2] - 1):
        # if i in range(start_idx - 1, end_idx):
        #     mask = masks_SA[:, :, i].copy()
        #     img = imgs_SA[:, :, i].copy()
        #     img = (img / np.max(img)) * 255
        #     np.clip(img, 0, 255, out=img)
        #

        # Segmentation of NN outliers

        # cv2_imshow(draw_image)

        # Segment LV
        img = imgs_SA[:, :, i].copy()
        img = (img / np.max(img)) * 255
        np.clip(img, 0, 255, out=img)
        img = img.astype('uint8')
        # cv2_imshow(img)

        filtered_img_LV = bandpass_filter(img.copy(), color1, color2)
        # cv2_imshow(filtered_img_LV)
        filtered_img_MYO = bandpass_filter(img.copy(), color1 * 1.2, color2, True)
        # cv2_imshow(filtered_img_MYO)
        save_img(filtered_img_LV, "A_filtered_img_LV_" + str(string.ascii_lowercase[i]))
        save_img(filtered_img_MYO, "B_filtered_img_MYO_" + str(string.ascii_lowercase[i]))

        th_LV = cv.adaptiveThreshold(filtered_img_LV, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 7)
        th_MYO = cv.adaptiveThreshold(filtered_img_MYO, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 7)
        # cv2_imshow(th_LV)
        # cv2_imshow(th_MYO)
        save_img(th_LV, "C_th_LV_" + str(string.ascii_lowercase[i]))
        save_img(th_MYO, "D_th_MYO_" + str(string.ascii_lowercase[i]))

        eroded_LV1 = cv.erode(th_LV, cv.getStructuringElement(cv.MORPH_ELLIPSE, (1, 1)))
        area_filtered_LV1 = filter_2D(eroded_LV1, area1, area2)

        # print("Eroded")
        eroded_LV2 = cv.erode(area_filtered_LV1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
                              iterations=erode_iterations)
        # cv2_imshow(eroded_LV2)
        area_filtered_LV2 = filter_2D(eroded_LV2, area1 * 2, area2 * 0.01)
        save_img(area_filtered_LV2, "E_area_filtered_LV_" + str(string.ascii_lowercase[i]))

        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(area_filtered_LV2, None, None, None, 4,
                                                                            cv.CV_32S)

        centroids = centroids[1:]
        LV_center = centroids[closest_node(c_LV, centroids)].copy()

        LV_label = labels[int(LV_center[1]), int(LV_center[0])]
        area = stats[1:, cv.CC_STAT_AREA][LV_label - 1]
        if area / area_nn < 0.2:
            dil_inc = 2
        else:
            dil_inc = 0

        LV_mask = np.zeros((labels.shape), np.uint8)
        LV_mask[labels == LV_label] = 255
        # cv2_imshow(LV_mask)
        save_img(LV_mask, "G_localized_LV_" + str(string.ascii_lowercase[i]))

        LV_mask = cv.dilate(LV_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
                            iterations=erode_iterations + 2 + dil_inc + LV_dilate_add)
        # cv2_imshow(LV_mask)
        save_img(LV_mask, "I_dilated_LV_" + str(string.ascii_lowercase[i]))

        # Segment MYO
        th_MYO = th_MYO + LV_mask
        th_MYO[th_MYO != 0] = 255
        # print("MYOO")
        # cv2_imshow(th_MYO)
        eroded_MYO1 = cv.erode(th_MYO, cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, 2)), iterations=2)
        area_filtered_MYO1 = filter_2D(eroded_MYO1, area1 * 0.5, area2)
        eroded_MYO2 = cv.erode(area_filtered_MYO1, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
                               iterations=erode_iterations + 3)
        # cv2_imshow(eroded_MYO2)
        area_filtered_MYO2 = filter_2D(eroded_MYO2, area1 * 2, area2 * 0.01)
        save_img(area_filtered_MYO2, "F_area_filtered_MYO_" + str(string.ascii_lowercase[i]))

        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(area_filtered_MYO2, None, None, None, 4,
                                                                            cv.CV_32S)
        centroids = centroids[1:]
        MYO_center = centroids[closest_node(LV_center, centroids)]
        draw_image = area_filtered_MYO2.copy()
        # print(MYO_center)
        cv.circle(draw_image, (int(MYO_center[0]), int(MYO_center[1])), 1, (0, 0, 0), 1)
        # cv2_imshow(draw_image)
        MYO_label = labels[int(MYO_center[1]), int(MYO_center[0])]

        MYO_mask = np.zeros((labels.shape), np.uint8)
        MYO_mask[labels == MYO_label] = 255
        # cv2_imshow(MYO_mask)
        save_img(MYO_mask, "H_localized_MYO_" + str(string.ascii_lowercase[i]))

        MYO_mask = cv.dilate(MYO_mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)),
                             iterations=erode_iterations + 5 + Myo_dilate_add)
        # print("FIN_RESULT")
        # cv2_imshow(MYO_mask)
        save_img(MYO_mask, "J_dilated_MYO_" + str(string.ascii_lowercase[i]))

        # Reasign labels
        LV_mask = Image.fromarray(LV_mask)
        MYO_mask = Image.fromarray(MYO_mask)
        LV_mask = np.array(LV_mask.filter(ImageFilter.ModeFilter(size=17)))
        MYO_mask = np.array(MYO_mask.filter(ImageFilter.ModeFilter(size=17)))

        # cv2_imshow(LV_mask)
        # cv2_imshow(MYO_mask)
        # print(np.max(MYO_mask))
        MYO_mask = MYO_mask - LV_mask

        LV_mask[LV_mask == 255] = 1
        MYO_mask[MYO_mask == 255] = 2

        mask = LV_mask + MYO_mask
        mask[mask == 3] = 2

        slice_locs_trimed.append(slice_locs[i])
        mask[mask == 1] = 75  # LV
        mask[mask == 2] = 150  # Myo
        mask[mask == 3] = 255  # RV
        # cv2_imshow(mask)

        LV_mask = mask.copy()
        LV_mask[LV_mask == 255] = 0
        LV_mask[LV_mask == 75] = 60
        LV_mask[LV_mask == 150] = 255

        mask_MYO = LV_mask.copy()
        mask_MYO[mask_MYO != 0] = 255
        np.clip(mask_MYO, 0, 255, out=mask_MYO)
        mask_MYO = mask_MYO.astype('uint8')
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(mask_MYO, None, None, None, 4, cv.CV_32S)
        areas = stats[1:, cv.CC_STAT_AREA]
        MYO_centers.append(centroids[-1])
        MYO_areas.append(areas[-1])

        # cv2_imshow(img)
        # cv2_imshow(mask)
        SA_mask_ED.append(mask)
        SA_img_ED.append(img)
        SA_LV_mask_ED.append(LV_mask)
        contoured_img = edge_detection(img, LV_mask)
        save_img(contoured_img, "K_contoured_img_" + str(string.ascii_lowercase[i]))

    MYO_centers = np.array(MYO_centers)
    MYO_areas = np.array(MYO_areas)

    SA_img_ED = np.array(SA_img_ED)
    slice_locs_trimed = np.array(slice_locs_trimed)

    SA_LV_mask_ED = np.array(SA_LV_mask_ED)

    # Delete outliers if any
    # print(SA_LV_mask_ED.shape)
    outliers_idx = (reject_outliers(MYO_centers[:, 0], outlier_threshold)) & (
        reject_outliers(MYO_centers[:, 1], outlier_threshold))
    outliers_idx2 = (reject_outliers(MYO_areas, outlier_threshold)) & (reject_outliers(MYO_areas, outlier_threshold))
    outliers_idx[idx_array] = True
    outliers_idx2[idx_array] = True

    SA_LV_mask_ED = SA_LV_mask_ED[(outliers_idx) & (outliers_idx2)]
    SA_img_ED = SA_img_ED[(outliers_idx) & (outliers_idx2)]
    slice_locs_trimed = slice_locs_trimed[(outliers_idx) & (outliers_idx2)]

    save_dicom(SA_LV_mask_ED, path, "ef_seg_sa_ED.nii.gz")
    save_dicom(SA_img_ED, path, "ef_sa_ED.nii.gz")
    save_gif(SA_LV_mask_ED, "sa_ed_seg_ef_gif/")
    save_gif(SA_img_ED, "sa_ed_ef_gif/")
    return SA_LV_mask_ED, SA_img_ED, slice_locs_trimed