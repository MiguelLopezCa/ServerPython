from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import json

# Descarga el modelo pre-entrenado en español
model = SentenceTransformer('hiiamsid/sentence_similarity_spanish_es')

app = Flask(__name__)

# Cargar los datos del JSON con codificación UTF-8
with open('señas.json', encoding='utf-8') as json_file:
    data = json.load(json_file)

def comparar_oraciones(oracion1, oracion2):
    # Genera los embeddings de las oraciones
    embedding1 = model.encode(oracion1, convert_to_tensor=True)
    embedding2 = model.encode(oracion2, convert_to_tensor=True)

    # Calcula la similitud coseno entre los embeddings
    similitud = util.pytorch_cos_sim(embedding1, embedding2)

    # Convierte la similitud en un valor entre 0 y 1
    similitud = similitud.item()

    return similitud

@app.route('/analizar', methods=['GET'])
def analizar_texto():
    texto = request.args.get('text')

    resultados = []

    embedding_texto = model.encode(texto, convert_to_tensor=True)

    for clave, valor in data.items():
        similitud = comparar_oraciones(clave, texto)
        resultados.append({'clave': clave, 'similitud': similitud})

    resultados = sorted(resultados, key=lambda x: x['similitud'], reverse=True)
    mejor_clave = resultados[0]['clave']
    similitud_maxima = resultados[0]['similitud']

    return jsonify({'mejor_clave': mejor_clave, 'similitud_maxima': similitud_maxima})

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=5000)


