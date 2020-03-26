# -*- coding: utf-8 -*-
import sys
import os
import time
import argparse
import json
import zipfile

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from PIL import Image
import cv2
from skimage import io
import numpy as np
from shapely.geometry import Polygon
import pandas as pd
from bs4 import BeautifulSoup
import requests

from craft_text_detection import craft_utils
from craft_text_detection import imgproc
from craft_text_detection import file_utils
from craft_text_detection.craft import CRAFT
from crnn.models import crnn
from crnn import utils, dataset
from collections import OrderedDict
from common_service.constants import *
from crnn.models import crnn


def calculate_area(bbox):
    polygon = Polygon(bbox)
    return polygon.area


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def test_net(net, image, text_threshold, link_threshold, low_text, cuda, poly, refine_net=None):
    # resize
    img_resized, target_ratio, size_heatmap = imgproc.resize_aspect_ratio(image, CANVAS_SIZE, interpolation=cv2.INTER_LINEAR, mag_ratio=MAG_RATIO)
    ratio_h = ratio_w = 1 / target_ratio

    # preprocessing
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1)    # [h, w, c] to [c, h, w]
    x = Variable(x.unsqueeze(0))                # [c, h, w] to [b, c, h, w]
    if cuda:
        x = x.cuda()

    # forward pass
    y, feature = net(x)

    # make score and link map
    score_text = y[0,:,:,0].cpu().data.numpy()
    score_link = y[0,:,:,1].cpu().data.numpy()

    # refine link
    if refine_net is not None:
        y_refiner = refine_net(y, feature)
        score_link = y_refiner[0,:,:,0].cpu().data.numpy()

    # Post-processing
    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, poly)

    # coordinate adjustment
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None: polys[k] = boxes[k]

    # render results (optional)
    render_img = score_text.copy()
    render_img = np.hstack((render_img, score_link))
    ret_score_text = imgproc.cvt2HeatmapImg(render_img)

    return boxes, polys, ret_score_text


def text_detection_service(image_list, cuda=False):
    # load net
    net = CRAFT()

    if cuda:
        net.load_state_dict(copyStateDict(torch.load(TRAINED_MODEL)))
    else:
        net.load_state_dict(copyStateDict(torch.load(TRAINED_MODEL, map_location='cpu')))

    if cuda:
        net = net.cuda()
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = False

    net.eval()

    # LinkRefiner
    refine_net = None
    POLY = False
    if REFINE:
        from refinenet import RefineNet
        refine_net = RefineNet()

        if cuda:
            refine_net.load_state_dict(copyStateDict(torch.load(REFINER_MODEL)))
            refine_net = refine_net.cuda()
            refine_net = torch.nn.DataParallel(refine_net)
        else:
            refine_net.load_state_dict(copyStateDict(torch.load(REFINER_MODEL, map_location='cpu')))

        refine_net.eval()
        POLY = True

    for k, image in enumerate(image_list):
        bboxes, polys, score_text = test_net(net, image, TEXT_THRESHOLD, LINK_THRESHOLD, LOW_TEXT, cuda, POLY, refine_net)
        max_area_index = [0, 0]
        max_area = [0, 0]

        for i in range(len(bboxes)):
            area = calculate_area(bboxes[i])

            if area >= max_area[0]:
                go_between = max_area_index[0]
                max_area_index[0] = i
                max_area_index[1] = go_between
                go_between = max_area[0]
                max_area[0] = area
                max_area[1] = go_between
            elif area >= max_area[1]:
                max_area_index[1] = i
                max_area[1] = area
            else:
                continue
        
        area_crops = []
        for i in range(len(max_area_index)):
            bbox_crop_ind = bboxes[max_area_index[i]]
            rect = cv2.boundingRect(bbox_crop_ind)
            x, y, w, h = rect
            y1 = y - 5 if y >=5 else 0
            y2 = y + h + 5
            x1 = x - 5 if x >=5 else 0
            x2 = x + w + 5
            area_crops.append(image[int(y1):int(y2), int(x1):int(x2)])
        return area_crops


def get_text_service(image):
    image = Image.fromarray(image).convert('L')
    model = crnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load(MODEL_PATH_CRNN))
    converter = utils.strLabelConverter(ALPHABET)
    transformer = dataset.resizeNormalize((100, 32))
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    model.eval()
    preds = model(image)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)
    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    
    return sim_pred


def get_side_effect(drug_name):
    drug_datas = pd.read_csv("static/csv/full_drug_info.csv")
    filter_elements = drug_datas[drug_datas["drug_name"].isin([drug_name])]
    side_effect_array = filter_elements["side_effect"].values

    return side_effect_array


def drug_elements(drug_name, path_file_full=PATH_FILE_FULL):
    drug_datas = pd.read_csv(path_file_full)
    filter_drugs = drug_datas.loc[(drug_datas["drug_name"] == drug_name.lower())]
    # import pdb; pdb.set_trace()
    # print(drug_datas.loc[(drug_datas["drug_name"] == drug_name)])

    # print(filter_drugs["drug_elements"])
    drug_elements_array = filter_drugs["drug_elements"].str.lower().values

    return drug_elements_array


def find_id_drugs_by_element(drug_element, path_file=PATH_FILE_ELEMENT):
    element_datas = pd.read_csv(path_file)
    # print(element_datas.shape)
    # import pdb; pdb.set_trace()
    filter_elements = element_datas.loc[(element_datas["drug_name"] == drug_element.capitalize())]
    # print(filter_elements["drug_name"])
    drugs_id_array = filter_elements["drug_id"].values[0]

    return drugs_id_array


def drug_interactions(drugs_id_array):
    inter_url = 'https://www.drugs.com/interactions-check.php?drug_list={}'.format(drugs_id_array)
    inter_r = requests.get(inter_url)
    inter_soup = BeautifulSoup(inter_r.text, 'html.parser')
    inter_between = inter_soup.find_all('div', 'interactions-reference')
    res_data = []
    for div in inter_between:
        interact_info = {}
        interact_info['level'] = div.find('span', 'ddc-status-label').text
        interact_info['title'] = div.find('h3').text.replace('  ', ' vs ')
        interact_info['apply'] = div.find('h3').findNext('p').text.split('Applies to:')[-1]
        interact_info['content'] = div.find('div', 'interactions-reference-header').findNext('p').findNext('p').text.replace('  ', ' ')
        res_data.append(interact_info)

    return res_data


def find_interaction(drug_names):
    for drug_name in drug_names:
        drug_elements_array = drug_elements(drug_name)
        drug_elements_array = ["Abacavir", "Abenol", "Adapalene", "Adagen"]
        drug_ids_array = []
        for drug_element_array in drug_elements_array:
            drug_id_array = find_id_drugs_by_element(drug_element_array)
            drug_ids_array.append(drug_id_array)

        drug_ids_array = np.array(drug_ids_array).flatten()
        drug_ids_array = list(set(drug_ids_array))
        drug_ids_array = np.array(drug_ids_array)
        interactions = drug_interactions("2-0,3557-0")

        return interactions
