from django.views.generic import TemplateView
from braces.views import JSONResponseMixin, AjaxResponseMixin
from PIL import Image
import numpy as np
import cv2 as cv
import pytesseract

from common_service.service import text_detection_service, get_text_service, get_side_effect, find_interaction
from common_service.ocr_pipelines import handle
from common_service.constants import TESERACT_CONFIG


class Conflict(TemplateView):
    template_name = "front-end/conflict.html"


class Sidefx(JSONResponseMixin, AjaxResponseMixin, TemplateView):
    template_name = "front-end/sidefx.html"

    def get_ajax(self, request, *args, **kwargs):
        drug_name = request.GET['drug_name']
        side_effects = get_side_effect(drug_name)
        side_effects = "Not found" if len(side_effects)==0 else side_effects[0]

        return self.render_json_response({"side_effects": side_effects}, 200)


class IndexView(JSONResponseMixin, AjaxResponseMixin, TemplateView):
    template_name = "front-end/index.html"

    def post_ajax(self, request, *args, **kwargs):
        img_temp = request.FILES["files"]
        input_img_type = request.POST["img_type"]
        
        text_recogs = []
        if (len(input_img_type) and input_img_type == "prescription"):
            area_crops = handle(img_temp)
            for i in range(len(area_crops)):
                text_recogs.append(pytesseract.image_to_string(area_crops[i], config=TESERACT_CONFIG))
        else:
            img = Image.open(img_temp)
            if not img.format == 'RGB':
                img = img.convert('RGB')

            img = np.asarray(img)
            area_crops = text_detection_service([img])
            for i in range(len(area_crops)):
                text_recogs.append(get_text_service(np.array(area_crops[i])).lower())
        interactions = find_interaction(text_recogs)
        interactions = 'Not found' if interactions == None or len(interactions) == 0 else interactions

        return self.render_json_response({"interactions": interactions}, 200)

    def get_context_data(self, **kwargs):
        context = super(IndexView, self).get_context_data(**kwargs)
        return context
