import pandas as pd

from scipy.interpolate import splprep, splev
from matplotlib import pyplot as plt

import numpy as np
import cv2 as cv
import cv2

tiff2 = []


def apply_mask(img, mask):
    np.clip(img, 0, 255, out=img)
    img = img.astype('uint8')

    myo_mask = np.where(mask == 255, 255, 0)
    np.clip(myo_mask, 0, 255, out=myo_mask)
    myo_mask = myo_mask.astype('uint8')
    rv_mask = np.where(mask == 120, 255, 0)
    np.clip(rv_mask, 0, 255, out=rv_mask)
    rv_mask = rv_mask.astype('uint8')
    lv_mask = np.where(mask == 60, 255, 0)
    np.clip(lv_mask, 0, 255, out=lv_mask)
    lv_mask = lv_mask.astype('uint8')

    MY = cv.bitwise_and(img.copy(), img.copy(), mask=myo_mask)
    RV = cv.bitwise_and(img.copy(), img.copy(), mask=rv_mask)
    LV = cv.bitwise_and(img.copy(), img.copy(), mask=lv_mask)
    full_mask = np.where(mask != 0, 255, 0)
    np.clip(full_mask, 0, 255, out=full_mask)
    full_mask = full_mask.astype('uint8')
    full = cv.bitwise_and(img.copy(), img.copy(), mask=full_mask)

    return LV, RV, MY, full


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


def reconstruction_full(img, mask, LV_mean, MY_mean):
    mask_copy = mask.copy()
    mask_copy2 = mask.copy()

    # Make areas less than LV holes
    mask_copy2[mask == 255] = 0
    mask_copy2[np.around(np.around(mask_copy2 / 10) * 10) == 60] = 255

    np.clip(mask_copy2, 0, 255, out=mask_copy2)
    mask_copy2 = mask_copy2.astype('uint8')
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_copy2, None, None, None, 4, cv2.CV_32S)
    areas = stats[1:, cv.CC_STAT_AREA]
    shape_max_label = np.argmax(areas) + 1
    for i in range(0, nlabels - 1):
        if i + 1 != shape_max_label:
            mask[labels == i + 1] = 0

    # Fill holes
    mask[mask != 0] = 255
    np.clip(mask, 0, 255, out=mask)
    mask = mask.astype('uint8')
    mask = cv2.bitwise_not(mask)
    holes = undesired_objects(mask, 100)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(holes, None, None, None, 4, cv2.CV_32S)
    for i in range(0, nlabels - 1):
        hole = np.zeros((labels.shape), np.uint8)
        hole[labels == i + 1] = 255
        hole_mean = np.mean(cv.bitwise_and(img.copy(), img.copy(), mask=hole))

        # If hole color closer to LV
        if np.abs(hole_mean - LV_mean) < np.abs(hole_mean - MY_mean):
            mask_copy[labels == i + 1] = 60
        else:
            mask_copy[labels == i + 1] = 255

    return mask_copy


def edge_detection(img):
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2_imshow(img_blur)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=0, threshold2=0)  # Canny Edge Detection

    contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    big_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(big_contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    result = np.zeros((img_blur.shape), np.uint8)
    cv2.drawContours(result, contours, -1, (255, 255, 255), thickness=1)

    # Display Canny Edge Detection Image
    # cv2_imshow(result)
    return result, [cX, cY]


def get_normal_vec(points):
    p1 = np.array(points[0])
    p2 = np.array(points[1])
    p3 = np.array(points[2])
    # These two vectors are in the plane
    v1 = p3 - p1
    v2 = p2 - p1
    # The cross product is a vector normal to the plane
    return np.cross(v1, v2)


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0] + 1)


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


from collections import OrderedDict


def extract_ring_path(indices, z_shape=False, spline=True):
    ring_indices = [[], []]
    x_idx_array = [[]]
    count = 0
    # Conpensate low resolution
    for i in range(len(indices[0])):
        x_array = indices[0]
        y_array = indices[1]

        x = x_array[i]
        y = y_array[i]
        if i in list(x_idx_array[0]):
            continue

        x_idx_array = np.where(x_array == x)
        y_idx_array = np.where(y_array == y)
        consecutive_y = consecutive(y_array[x_idx_array])
        # If straight line
        if len(consecutive_y) == 1:
            ring_indices[0].extend(indices[0][x_idx_array])
            ring_indices[1].extend(indices[1][x_idx_array])

        # Seperate groupes alone y
        if len(consecutive_y) >= 2:
            ring_indices[0].extend(indices[0][x_idx_array])
            for j in range(len(consecutive_y)):
                group_len = len(consecutive_y[j])
                ring_indices[1].extend([np.mean(consecutive_y[j])] * group_len)

    ring_coords = list(zip(ring_indices[0], ring_indices[1]))
    ring_coords = np.array(list(OrderedDict.fromkeys(ring_coords)))

    ring_df = pd.DataFrame(data=ring_coords, columns=['x', 'y'])

    unique_x_vals = ring_df["x"].unique()

    ordered_ring_coords = []
    mirror_coords = []
    count = 0
    for unique_x in unique_x_vals:
        points_at_x = ring_df[ring_df["x"] == unique_x]
        # display(points_at_x)
        conseq_y_bool = points_at_x["y"].diff().eq(1).any()
        if conseq_y_bool and count == 0:
            ordered_ring_coords.extend(np.array(points_at_x))
        else:
            points_at_x_array = np.array(points_at_x)

            ordered_ring_coords.extend(points_at_x_array[1:][::-1])
            mirror_coords.append(points_at_x_array[0])
        count += 1
    mirror_coords = mirror_coords[::-1]
    mirror_coords.append(ordered_ring_coords[0])
    ordered_ring_coords.extend(mirror_coords)

    # center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), ring_coords), [len(ring_coords)] * 2))
    # ordered_ring_coords = sorted(ring_coords, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360)
    # ordered_ring_coords = np.array(ordered_ring_coords)

    # plt.figure()
    # plt.plot(indices[0], indices[1], "ro")
    # plt.plot(indices[0], indices[1])
    #
    # plt.figure()
    # plt.plot(ring_indices[0], ring_indices[1], "ro")
    # plt.plot(ring_indices[0], ring_indices[1])
    #
    # plt.figure()
    # plt.plot(*zip(*ordered_ring_coords), "ro")
    # plt.plot(*zip(*ordered_ring_coords))
    x_new = list(list(zip(*ordered_ring_coords))[0])
    y_new = list(list(zip(*ordered_ring_coords))[1])
    if spline:
        ordered_ring_coords = np.array(ordered_ring_coords)
        tck, u = splprep(ordered_ring_coords.T, u=None, s=0.0, per=1)
        u_new = np.linspace(u.min(), u.max(), 200)
        x_new, y_new = splev(u_new, tck, der=0)
        if z_shape:
            print(u_new)
        # plt.figure()
        # plt.plot(ordered_ring_coords[:, 0], ordered_ring_coords[:, 1], "ro")
        # plt.plot(x_new, y_new, 'b-')

    return x_new, y_new


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')


def generate_xzy(indices, z_depth, center_diff_x=0, center_diff_y=0, scatter=True):
    coordinates = np.array(
        [indices[0] + center_diff_x, indices[1] + center_diff_y, np.array(np.shape(indices)[1] * [z_depth])])

    # if scatter:
    #     ax.scatter(coordinates[0], coordinates[1], coordinates[2])
    # else:
    #     ax.plot(coordinates[0], coordinates[1], coordinates[2], "r-")

    coordinates_zipped = list(zip(coordinates[0], coordinates[1], coordinates[2]))

    coordinates = np.array(
        [indices[0], indices[1], np.array(np.shape(indices)[1] * [z_depth]), np.array(np.shape(indices)[1] * [255]),
         np.array(np.shape(indices)[1] * [255]), np.array(np.shape(indices)[1] * [255])])

    coordinates_zipped = list(
        zip(coordinates[0], coordinates[1], coordinates[2], coordinates[3], coordinates[4], coordinates[5]))
    return coordinates_zipped


def save_xzys(xyz_array, folder, filename):
    xyz_array = [item for sublist in xyz_array for item in sublist]
    xyz_array_np = []

    xyz_array_np = np.array(xyz_array)
    np.savetxt(folder + filename, xyz_array_np, delimiter=',')


def reconstruct_tip(edge, num_img, range=[0.9, 0.01]):
    sizes = np.linspace(range[0], range[1], num=num_img)
    l_dim = np.shape(edge)[0]

    reconstructed_shapes = []
    for size in sizes:
        small = cv2.resize(edge, (0, 0), fx=size, fy=size)
        s_dim = np.shape(small)[0]

        x_offset = y_offset = round((l_dim - s_dim) / 2)

        result = np.zeros((edge.shape), np.uint8)
        result[y_offset:y_offset + small.shape[0], x_offset:x_offset + small.shape[1]] = small
        # cv2_imshow(small)
        # cv2_imshow(result)
        reconstructed_shapes.append(result)
    return reconstructed_shapes


def extract_contour(SA_LV_mask_ED, SA_img_ED, slice_locs_trimed, folder):
    if len(SA_LV_mask_ED)>5:
        SA_LV_mask_ED = SA_LV_mask_ED[1:-1]
        SA_img_ED = SA_img_ED[1:-1]
        slice_locs_trimed = slice_locs_trimed[1:-1]


    xyz_array_myo = []
    xyz_array_lv = []
    SA_LV_mask_ED_copy = SA_LV_mask_ED.copy()

    full_arr = []
    z_depth1 = slice_locs_trimed[0]
    z_depth2 = slice_locs_trimed[0]

    for i in range(len(SA_LV_mask_ED)):
        # cv2_imshow(tiff_copy[i])
        reconstructed = SA_LV_mask_ED_copy[i]
        tiff2.append(reconstructed)
        np.clip(reconstructed, 0, 255, out=reconstructed)
        reconstructed = reconstructed.astype('uint8')
        _, _, _, centroids = cv2.connectedComponentsWithStats(reconstructed, None, None, None, 4, cv2.CV_32S)

        LV, RV, MY, full = apply_mask(SA_img_ED[i], reconstructed)
        full_arr.append(full)
        # cv2_imshow(reconstructed)
        # cv2_imshow(full)
        myo_solid = reconstructed.copy()
        myo_solid[myo_solid != 0] = 255
        edges, _ = edge_detection(myo_solid)

        LV_solid = reconstructed.copy()
        LV_solid[LV_solid == 255] = 0
        LV_solid[LV_solid != 0] = 255
        edges2, _ = edge_detection(LV_solid)

        indices = np.array(np.where(edges != [0]))
        ring_x, ring_y = extract_ring_path(indices)
        xyz_array_myo.append(generate_xzy([ring_x, ring_y], z_depth1, scatter=False))
        try:
            z_depth1 = slice_locs_trimed[i + 1]
        except:
            pass

        indices = np.array(np.where(edges2 != [0]))
        ring_x, ring_y = extract_ring_path(indices)
        xyz_array_lv.append(generate_xzy([ring_x, ring_y], z_depth2, scatter=False))
        try:
            z_depth2 = slice_locs_trimed[i + 1]
        except:
            pass

    save_xzys(xyz_array_myo, folder, "Myo_point_cloud.xyz")
    save_xzys(xyz_array_lv, folder, "LV_point_cloud.xyz")
