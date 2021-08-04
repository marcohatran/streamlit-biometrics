import cv2
import numpy as np
from service.preprocess import preprocess
import service.features as features
import service.helper as helper
import os, io, csv
import sys
from service.helper import shiftcorrection, cropfingerprint, find_roi
import scipy
from skimage.morphology import skeletonize, thin
from sklearn.neighbors import NearestNeighbors
from service.lineimitator import createLineIterator
import service.constants as const



def convert_to_polar(obj, index):
    base = obj[index]
    polars = []
    polar_obj = []
    for current in obj:
        cur = features.FeaturePolar(current, base)
        polars.append(cur.convert())
        polar_obj.append(cur)
    return polars, polar_obj


def get_most_similar(fv1, fv2):
    s = 0
    x, y = None, None
    for i in range(len(fv1)):
        for j in range(len(fv2)):
            si = helper.similarity(fv1[i], fv2[j])

            if si > s:
                x = i
                y = j
                s = si

    return x, y, s


def get_ridge_count(rcoords, image):
    mask = np.zeros(image.shape)
    coords = np.array([[int(i), int(j)] for i, j, k in rcoords])
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='ball_tree').fit(coords)
    distances, indices = nbrs.kneighbors(coords)
    ridgecount = []
    for i in range(len(coords)):
        p1, p2 = np.array(coords[indices[i][1]]), np.array(coords[indices[i][2]])
        p1_type, p2_type = rcoords[indices[i][1]][2], rcoords[indices[i][2]][2]
        r = np.array(coords[i])
        iter1 = createLineIterator(r, p1, image)
        iter2 = createLineIterator(r, p2, image)
        mask[coords[i][0], coords[i][1]] = 1

        l1 = [image[int(x), int(y)] for x, y, z, in iter1]

        for i in range(len(l1)):
            if l1[i] != 1:
                l1 = l1[i:]
                break

        while l1 and l1[-1] == 1:
            l1.pop()

        l2 = [image[int(x), int(y)] for x, y, z, in iter2]

        for i in range(len(l2)):
            if l2[i] != 1:
                l2 = l2[i:]
                break

        while l2 and l2[-1] == 1:
            l2.pop()

        l1_ = [l1[i].astype(int) - l1[i + 1].astype(int) for i in range(len(l1) - 1)]
        l2_ = [l2[i].astype(int) - l2[i + 1].astype(int) for i in range(len(l2) - 1)]
        l1_c = 0
        l2_c = 0

        trail = True
        sum = 0
        for i in l1_:
            sum = sum + i
            if sum == 0 and trail == False:
                l1_c = l1_c + 1
                trail = True
            elif (sum != 0):
                trail = False

        trail = True
        sum = 0
        for i in l2_:
            sum = sum + i
            if sum == 0 and trail == False:
                l2_c = l2_c + 1
                trail = True
            elif (sum != 0):
                trail = False

        # print(l1_c, l2_c)
        # print(l1_)
        # print(l2_)
        to_add = [p1, p1_type, l1_c, p2, p2_type, l2_c]
        ridgecount.append(to_add)
    # cv2.imwrite("thinnedimage0.jpg", mask*255)
    # break
    # print ridgecount

    return ridgecount


def extract_minutiae(image: np.array):
    """
    Crossing number technique for minutiae extraction from skeletonised binarised images
    Based on http://airccse.org/journal/ijcseit/papers/2312ijcseit01.pdf
    Requires binarised image array with integer values in [0, 1]. Where 1 is ridge.

    Args:
        image (np.array): Image as a np array - 1 channel gray-scale, with white background

    Returns:
        list: [terminations, bifurcations] - extracted from the given image.
                    terminations (list) - tuple coordinates for the location of a ridge termination
                    bifurcations (list) - tuple coordinates for the location of a ridge bifurcation

    """

    # Index order list - defines the order in which the pixels in a 3x3 frame are considered.
    idx = [(1, -1), (0, -1), (0, 1), (0, 0), (1, 0), (-1, 0), (-1, 1), (-1, -1), (1, -1)]

    debug = False

    height, width = image.shape
    mask = np.zeros(image.shape)

    # Store all minutiae
    bifurcations = []
    terminations = []

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # 3x3 frame extraction based on the previous, current and next values on x and y axis.
            frame = image[i - 1: i + 2, j - 1: j + 2]

            # Custom minutiae detection function.
            # Control for pixels found in the middle of the frame.
            # Once identified, it counts filled pixels separated by at least 1 empty pixel.
            pixel_list = [frame[idx[i]] * (1 - frame[idx[i + 1]]) for i in range(len(idx) - 1)]
            pixel_sum = frame[1, 1] * sum(pixel_list)

            # Based on http://airccse.org/journal/ijcseit/papers/2312ijcseit01.pdf
            # pixel_sum = .5 * sum([abs(frame[idx[i]] - frame[idx[i + 1]]) for i in range(len(indices) - 1)])

            if pixel_sum == 1:
                # Termination

                # Add termination coordinates
                terminations.append((i, j, 1))
                mask[i, j] = 1

            elif pixel_sum == 3:
                # Bifurcation

                # Add bifurcation coordinates
                bifurcations.append((i, j, 3))
                mask[i, j] = 1

    return terminations, bifurcations, mask


def clean_minutiae(image: np.array, minutiae: list) -> list:
    """
    Post-processing
    Remove minutiae identified on the outer terminations of the image.
    We identify outer minutiae as follows: For each type of minutia, we check its quadrant.
    If there are no other full pixels to both the closest sides to an edge on both x and y coord
    That minutiae is discraded.
    Checks location and other pixel values to the sides of the minutiae.
    Outputs list of cleaned minutiae.
    Args:
        image (np.array): Image to be analysed for cleaning borderline minutiae.
        minutiae  (list): Minutiae represented as a list of coordinate tuples (2d: x, y))
    Returns:
        list: Coordinate as tuple list of minutiae that are not found at the image bordering ridge terminations.
    """

    height, width = image.shape

    minutiae_clean = []
    for x, y, k in minutiae:
        # If there are directions in which the minutiae with x and y coordinates has only empty
        # pixels, that we label the minutiae as an image border and discard it.
        if (image[x, :y].sum() > 0) and (image[x, y + 1:].sum() > 0) and (image[:x, y].sum() > 0) and \
                (image[x + 1:, y].sum() > 0):
            minutiae_clean.append((x, y, k))

    return minutiae_clean


def process_minutiae(image: np.array):
    """
    Image processing into minutiae - bifurcations
    Args:
        image   (np.array): Image in 1 channel gray-scale.
    Returns:
        list:     minutiae list containing minutiae coordinates (x, y)
    """

    # Extract minutiae
    terminations, bifurcations, mask = extract_minutiae(image)

    # Post-processing border minutiae removal.
    terminations = clean_minutiae(image, terminations)
    bifurcations = clean_minutiae(image, bifurcations)

    return terminations + bifurcations, mask


def removedot(invertThin):
    temp0 = np.array(invertThin[:])
    temp0 = np.array(temp0)
    temp1 = temp0 / 255
    temp2 = np.array(temp1)
    temp3 = np.array(temp2)

    enhanced_img = np.array(temp0)
    filter0 = np.zeros((10, 10))
    W, H = temp0.shape[:2]
    filtersize = 6

    for i in range(W - filtersize):
        for j in range(H - filtersize):
            filter0 = temp1[i:i + filtersize, j:j + filtersize]

            flag = 0
            if sum(filter0[:, 0]) == 0:
                flag += 1
            if sum(filter0[:, filtersize - 1]) == 0:
                flag += 1
            if sum(filter0[0, :]) == 0:
                flag += 1
            if sum(filter0[filtersize - 1, :]) == 0:
                flag += 1
            if flag > 3:
                temp2[i:i + filtersize, j:j + filtersize] = np.zeros((filtersize, filtersize))

    return temp2


def match_level(pv1, pv2, fv1, fv2):
    ml = np.zeros((len(pv1), len(pv2)))

    for i in range((len(pv1))):
        for j in range((len(pv2))):
            if np.all(np.abs(pv1[i] - pv2[j]) > const.BG):
                continue

            ml[i, j] = 0.5 + (0.5 * helper.similarity(fv1[i], fv2[j]))

    # ml_prime = np.zeros((len(pv1),len(pv2)))

    # for i,row in enumerate(ml):
    # 	j = np.argmax(row)
    # 	ml_prime[i,j] = row[j]

    # ml = ml_prime
    # ml_prime = np.zeros((len(pv1),len(pv2)))

    # for j,col in enumerate(ml.T):
    # 	i = np.argmax(col)
    # 	ml_prime[i,j] = col[i]

    # print(ml_prime)
    sum = 0
    count = 0
    while ml.any() != 0:
        ind = np.argmax(ml)
        x, y = ind // len(pv2), ind % (len(pv2))
        if ml[x, y] != 0.5: sum = sum + ml[x, y]
        # sum = sum + ml[x,y]
        # print(ml[x,y])
        ml[x] = 0
        ml[:, y] = 0
        count = count + 1

    return (sum / count)


def R2(img2):
    kernal = np.ones((3, 3), np.uint8)
    image2 = helper.Roi(img2)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    image2 = cv2.morphologyEx(image2, cv2.MORPH_OPEN, kernal, iterations=1)
    image2, m2, orientations2 = preprocess(image2)
    for i in range(image2.shape[0]):
        for j in range(image2.shape[1]):
            if image2[i][j] > 50:
                image2[i][j] = 1
            else:
                image2[i][j] = 0
    image2, xmax, xmin, ymax, ymin = cropfingerprint(image2)
    # cv2.imwrite(dir + "crpped_img_2.jpg", image2*255)
    orientations2 = orientations2[xmin:xmax + 1, ymin:ymax + 1]
    # cv2.imwrite(dir + "intermediate-input2.jpg", image2*255)
    skeleton = skeletonize(image2)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)
    thinned = thin(skeleton)
    thinned2 = np.array(thinned, dtype=np.uint8)
    return thinned2, orientations2


def get_matches(img1, img2, orientations2):
    kernal = np.ones((3, 3), np.uint8)
    image1 = helper.Roi(img1)
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(dir + "Roi1.jpg", image1)
    # plt.imshow(image1)
    # plt.show()

    ''' Image 1 Processing from here '''
    image1 = cv2.morphologyEx(image1, cv2.MORPH_OPEN, kernal, iterations=1)
    # plt.imshow(image1)
    # plt.show()
    image1, m1, orientations1 = preprocess(image1)
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            if image1[i][j] > 50:
                image1[i][j] = 1
            else:
                image1[i][j] = 0
    image1, xmax, xmin, ymax, ymin = cropfingerprint(image1)
    # plt.imshow(image1*255)
    # plt.show()
    orientations1 = orientations1[xmin:xmax + 1, ymin:ymax + 1]
    # cv2.imwrite(dir + "intermediate-input1.jpg", image1*255)
    skeleton = skeletonize(image1)
    skeleton = np.array(skeleton, dtype=np.uint8)
    skeleton = removedot(skeleton)
    thinned1 = thin(skeleton)
    thinned1 = np.array(thinned1, dtype=np.uint8)
    # cv2.imwrite(dir + "thinnedimage-input1.jpg", (1-thinned1)*255)
    fincoords1, mask1 = process_minutiae(thinned1)
    # cv2.imwrite(dir + "input_minute.jpg", mask1*255)
    vector1 = get_ridge_count(fincoords1, thinned1)
    print(vector1)
    # print(len(fincoords1))
    fv1, fo1 = features.get_features(fincoords1, vector1, orientations1)

    ''' Image 2 processing '''
    thinned2 = img2
    # cv2.imwrite(dir + "thinnedimage-input2.jpg", (1-thinned2)*255)
    fincoords2, mask2 = process_minutiae(thinned2)
    # cv2.imwrite(dir + "input_minute2.jpg", mask2*255)
    vector2 = get_ridge_count(fincoords2, thinned2)
    print(vector2)
    # print(len(fincoords2))
    fv2, fo2 = features.get_features(fincoords2, vector2, orientations2)

    sl = get_most_similar(fv1, fv2)
    b1 = sl[0]
    b2 = sl[1]
    pv1, po1 = convert_to_polar(fo1, b1)
    pv2, po2 = convert_to_polar(fo2, b2)
    ml = match_level(pv1, pv2, fv1, fv2)
    print(ml, "Matched" if (ml > 0.2) else "Not Matched")
    if ml > 0.2:
        return {"Result": True, "Score": ml}
    else:
        return {"Result": False, "Score": ml}




def compare(img_path1, img_path2):
    img2 = cv2.imread(img_path1, cv2.IMREAD_COLOR)
    thin2, orientations2 = R2(img2)
    img1 = cv2.imread(img_path2, cv2.IMREAD_COLOR)
    data = get_matches(img1, thin2, orientations2)
    os.remove(img_path1)
    os.remove(img_path2)
    return data
