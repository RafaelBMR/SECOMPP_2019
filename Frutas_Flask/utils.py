import numpy as np
import cv2
import json

from keras.layers import Input, Dense, Conv2D, SpatialDropout2D, MaxPooling2D, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def predicao(modelo, imagem):
    imagem_array = cv2.imread(imagem)
    with open("fruta2id.json", mode="r") as f:
        fruta2id = json.load(f)
    predicao = modelo.predict(imagem_array.reshape(1, 100, 100, 3))
    id2fruta = {v: k for k, v in fruta2id.items()}
    x_probabilidades = []
    y_probabilidades = []
    for i in range(len(id2fruta)):
        x_probabilidades.append(id2fruta[i][:8])
        y_probabilidades.append(predicao[0][i])

    fig = plt.figure(figsize=(12,6))
    ax = sns.barplot(x_probabilidades, y_probabilidades)
    fig.add_axes(ax)
    sio = io.BytesIO()
    fig.savefig(sio, format="png")
    img_probabilidades_codificada = base64.b64encode(sio.getvalue())
    img_probabilidades = '<img src="data:image/png;base64,%s" />' % img_probabilidades_codificada.decode('utf-8')
    fig.clf()
    del fig, ax

    return img_probabilidades

def saida_conv1(modelo, imagem):
    imagem_array = cv2.imread(imagem)
    entrada = Input(shape=(100, 100, 3))
    conv1 = Conv2D(16, 2, activation='relu', padding='same')(entrada)
    max1 = MaxPooling2D()(conv1)
    conv2 = Conv2D(32, 2, activation='relu', padding='same')(max1)
    max2 = MaxPooling2D()(conv2)
    conv3 = Conv2D(64, 2, activation='relu', padding='same')(max2)
    max3 = MaxPooling2D()(conv3)
    conv4 = Conv2D(128, 2, activation='relu', padding='same')(max3)
    max4 = MaxPooling2D()(conv4)
    spatial_drop = SpatialDropout2D(0.3)(max4)
    flatten = Flatten()(spatial_drop)
    densa = Dense(150, activation='relu')(flatten)
    densa_drop = Dropout(0.4)(densa)
    saida = Dense(14, activation='softmax')(densa_drop)
    model = Model(entrada, conv1)
    model.set_weights(modelo.get_weights())
    saida_conv = model.predict(imagem_array.reshape(1,100,100,3))[0]
    graficos = []
    for i in range(16):
        fig = plt.figure(figsize=(5,5))
        ax = sns.heatmap(saida_conv[:, :, i], cmap='Blues', center=0.0)
        fig.add_axes(ax)
        sio = io.BytesIO()
        fig.savefig(sio, format="png", bbox_inches="tight")
        grafico_codificado = base64.b64encode(sio.getvalue())
        plt.clf()
        del fig, ax
        grafico_html = '<img src="data:image/png;base64,%s" />' % grafico_codificado.decode('utf-8')
        graficos.append(grafico_html)

    return graficos



