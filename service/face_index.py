import face_recognition.api as face_recognition
from elasticsearch import Elasticsearch
import json
import os

def face_encoding(image_path):
    image = face_recognition.load_image_file(image_path)
    face_vector = face_recognition.face_encodings(image)
    return face_vector[0].tolist()
def face_index(face_vector, cif, phone):
    host = "localhost"
    port = "9200"
    index = "faces_demo"
    es = Elasticsearch()
    data = json.dumps({
        "face_encoding": face_vector,
        "cif": str(cif),
        "phone": str(phone)
    })
    try:
        es.index(index=str(index), body=data)
        return "Success"
    except Exception as e:
        return "Failure"
