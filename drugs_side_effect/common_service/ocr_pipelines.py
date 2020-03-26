import cv2
import numpy as np
import pytesseract
from PIL import Image


def preprocess_image(img):
    """
    img: gray image, np.array
    """
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    return thresh


def mask_vertical_horizal(img, thresh):
    """
    img: origin image
    thresh: thresh image
    """
    horizal = thresh
    vertical = thresh

    scale_height = 20
    scale_long = 15

    long = int(img.shape[1] / scale_long)
    height = int(img.shape[0] / scale_height)

    horizalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (long, 1))
    horizal = cv2.erode(horizal, horizalStructure, (-1, -1))
    horizal = cv2.dilate(horizal, horizalStructure, (-1, -1))

    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height))
    vertical = cv2.erode(vertical, verticalStructure, (-1, -1))
    vertical = cv2.dilate(vertical, verticalStructure, (-1, -1))

    mask = vertical + horizal
    return mask


def get_largest_contour_coords(mask, px=5):
    """
    contour of mask image
    """
    _, contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_ = -1
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > max_:
            x_max, y_max, w_max, h_max = x, y, w, h
            max_ = cv2.contourArea(cnt)

    x_max -= px
    y_max -= px
    w_max += px * 2
    h_max += px * 2

    return x_max, y_max, w_max, h_max


def preprocess_table(mask_table):
    kernel = np.ones((3, 3), np.uint8)
    table_thresh = cv2.dilate(mask_table, kernel, iterations=1)
    return table_thresh


def find_text_boxes(pre, min_text_height_limit=6, max_text_height_limit=40):
    """
    find all text boxes of table-thresh after preprocessing
    """
    _, contours, hierarchy = cv2.findContours(
        pre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for contour in contours:
        box = cv2.boundingRect(contour)
        # h = box[3]
        # if min_text_height_limit < h < max_text_height_limit:
        #     boxes.append(box)
        boxes.append(box)

    return boxes


def get_all_drug_name_imgs(table, boxes, no_rows):
    no_rows = len(boxes) // 7  # fixed no columns
    sorted_boxes = sorted(boxes, key=lambda x: x[0])

    # get all text boxes contain drug names
    # exclude 1st with largest area
    text_boxes = sorted_boxes[1:][1 * no_rows:2 * no_rows]

    # sorted text boxes by row
    text_boxes = sorted(text_boxes, key=lambda x: x[1])[1:]

    text_imgs = []
    for box in text_boxes:
        # x, y, w, h
        sub_img = table[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
        text_imgs.append(sub_img)

    return text_imgs


def handle(img_fp):
    ori_img = Image.open(img_fp).convert('L')
    ori_img = np.asarray(ori_img)
    thresh = preprocess_image(ori_img)
    mask = mask_vertical_horizal(ori_img, thresh)
    x_max, y_max, w_max, h_max = get_largest_contour_coords(mask)
    table = ori_img[y_max:y_max + h_max, x_max:x_max + w_max]
    mask_table = mask[y_max:y_max + h_max, x_max:x_max + w_max]

    table_thresh = preprocess_table(mask_table)

    boxes = find_text_boxes(table_thresh)
    no_rows = len(boxes) // 7  # fixed columns

    sub_imgs = get_all_drug_name_imgs(table, boxes, no_rows)
    return sub_imgs
