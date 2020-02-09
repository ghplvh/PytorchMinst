from flask import Flask, render_template, request
import numpy as np
import torch
import re
import base64
from torchvision import transforms
from PIL import Image
from model.cnn_model import MNISTCnn
from torch.autograd import Variable

app = Flask(__name__)

model_file = './model/model.h5'
model = torch.load(model_file)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict/', methods=['Get', 'POST'])
def preditc():
    parseImage(request.get_data())
    img = to_transform()
    response = predict(img)

    return response


def parseImage(imgData):
    """
    生成图片
    :param imgData:
    :return:
    """
    imgStr = re.search(b'base64,(.*)', imgData).group(1)
    with open('./output.png', 'wb') as output:
        output.write(base64.decodebytes(imgStr))


def to_transform():
    """
    将图片转换为28*28尺寸的灰度图并归一化
    # 增加一个维度 1 28 28
    :return:
    """
    img = Image.open('output.png')
    df_transforms = transforms.Compose(
        [transforms.Resize((28, 28)), transforms.Grayscale(num_output_channels=1), transforms.ToTensor()])
    copy_img = df_transforms(img)
    copy_img = copy_img.unsqueeze(0)
    return copy_img


def predict(img):
    """
    预测数字
    :return:
    """
    out = model(img)

    _, pred = torch.max(out, 1)
    # 转换张量tensor([4])为numpy数组得到一般数字
    resp = pred[0].numpy()

    return str(resp)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8888)
