import base64
import dataclasses
import json
import math
import os.path
import time
from io import BytesIO

import cv2
import numpy as np
import torch
from PIL import Image
from django.http import HttpResponse, HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from net.model1 import load_model as load_model1
from net.model_simple_conv import load_module as load_module_simple_conv
from net.model_simple_conv_mnist import load_module as load_module_simple_conv_mnist
from net.model_le_net5 import load_module as load_module_le_net5
from .reg import reg


def index(request):
    return HttpResponse("Hello!!")


def base64_to_image(base64_data):
    # 从Base64解码为二进制数据
    image_data = base64.b64decode(base64_data.split(",")[1])
    image = Image.open(BytesIO(image_data))
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.asarray(img)
    return img


@dataclasses.dataclass()
class Model:
    model: torch.nn.Module
    accuracy: float
    name: str
    desc: str


models = [Model(load_model1(), 0.78, 'Affine单层网络', ''),
          Model(load_module_simple_conv(), 0.98, '多层简单CNN网络',
                'Conv(30, 5, 5) -> Relu -> MaxPooling -> Affine -> Relu -> Affine'),
          Model(load_module_le_net5(), 0.94, 'LeNet5', ''),
          # Model(load_module_simple_conv_mnist(), 0.94, '多层简单CNN网络(mixed data)',
          #       '和上面的一样，只是加入了mnist数据训练'),
          ]


def list_model(request: HttpRequest):
    # pass
    return JsonResponse({
        'code': 0,
        'msg': '',
        'data': [{'name': m.name, 'accuracy': m.accuracy, 'desc': m.desc} for m in models]
    })


@csrf_exempt
def create(request: HttpRequest):
    json_data = json.loads(request.body)
    img = json_data['img']
    num = json_data['num']
    if not img or num is None:
        return JsonResponse({
            'code': 1,
            'msg': 'img or num is empty'
        })
    img = base64_to_image(img)
    dst_file = os.path.join('runtime', 'dataset', f'{math.floor(time.time() * 1000)}_{num}.jpg')
    if not os.path.exists(os.path.dirname(dst_file)):
        os.mkdir(os.path.dirname(dst_file))
    cv2.imwrite(dst_file, img)
    return JsonResponse({
        'code': 0,
        'msg': ''
    })


@csrf_exempt
def recognize(request: HttpRequest):
    json_data = json.loads(request.body)
    img = json_data['img']
    img = base64_to_image(img)
    model_name = json_data['model']
    # save_recognize_img(img)

    if not model_name:
        m = models[0]
    else:
        m = next((m for m in models if m.name == model_name), None)
    if m is None:
        return JsonResponse({
            'code': 1,
            'msg': f"can't find model named {model_name}"
        })
    print('m: ', m)
    m = m.model

    rst = reg(img, m)[0]
    rst = np.round(rst, 2)
    np.set_printoptions(precision=2)

    formatted_rst = [f'{value:.2f}' for value in rst]
    return JsonResponse({
        'code': 0,
        'msg': '',
        'data': {
            'possible': formatted_rst,
            'rst': int(np.argmax(rst))
        }
    })


def save_recognize_img(img):
    dst_file = os.path.join('runtime', f'{math.floor(time.time() * 1000)}.jpg')
    if not os.path.exists(os.path.dirname(dst_file)):
        os.mkdir(os.path.dirname(dst_file))
    cv2.imwrite(dst_file, img)
