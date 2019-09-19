from flask import Flask, render_template
import os
from flask import flash, request, redirect, url_for

from utils import predicao, saida_conv1

from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

modelo = load_model("model_frutas_20.hdf5")
graph = tf.get_default_graph()

@app.route('/', methods=['GET', 'POST'])
def index():
    probabilidades = {}
    saida_conv = None
    if request.method == "POST":
        global graph
        caminho_imagem = request.form['inputArea']
        with graph.as_default():            
            probabilidades = predicao(modelo, caminho_imagem)
        with graph.as_default():
            saida_conv = saida_conv1(modelo, caminho_imagem)
    return render_template("index.html", data={'probabilidades': probabilidades, 'saida_conv': saida_conv})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)