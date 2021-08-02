import os
import streamlit as st
from PIL import Image
import PIL
import uuid
from service.face_compare import main
from service.face_index import face_encoding, face_index
from service.face_search import face_search

# Create a page dropdown
title = st.title("Nhận diện sinh trắc học")
page = st.selectbox("Lựa chọn chức năng", ["Face Compare", "Face Index", "Face Search"])
if page == "Face Compare":
    upload_face = st.file_uploader("Chọn ảnh khuôn mặt ...")
    if upload_face is not None:
        image_face = Image.open(upload_face)
        st.image(image_face, caption="Uploaded", use_column_width=True)
        image_face_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_face.save(image_face_path)
    upload_idcard_front = st.file_uploader("Chọn ảnh căn cước công dân mặt trước ...")
    if upload_idcard_front is not None:
        image_idcard_front = Image.open(upload_idcard_front)
        st.image(image_idcard_front, caption="Uploaded", use_column_width=True)
        image_idcard_front_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_idcard_front.save(image_idcard_front_path)
    if st.button('Face compare'):
        distance, result = main(image_face_path, image_idcard_front_path)
        st.write("Kết quả:" + str(result))
        st.write("Tỷ lệ: " + str(1 - distance))
        os.remove(image_face_path)
        os.remove(image_idcard_front_path)
elif page == "Face Index":
    upload_face_index = st.file_uploader("Chọn ảnh khuôn mặt ...")
    if upload_face_index is not None:
        image_to_index = Image.open(upload_face_index)
        st.image(image_to_index, caption="Uploaded", use_column_width=True)
        image_to_index_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_to_index.save(image_to_index_path)
    cif = st.text_input("Nhập số cif")
    phone = st.text_input("Nhập số điện thoại")
    if st.button("Index Face"):
        face_vector = face_encoding(image_to_index_path)
        result = face_index(face_vector, cif, phone)
        st.write(result)
        os.remove(image_to_index_path)


elif page == "Face Search":
    upload_face_search = st.file_uploader("Chọn ảnh khuôn mặt ...")
    if upload_face_search is not None:
        image_to_search = Image.open(upload_face_search)
        st.image(image_to_search, caption="Uploaded", use_column_width=True)
        image_to_search_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_to_search.save(image_to_search_path)
    if st.button("Search"):
        face_vector = face_encoding(image_to_search_path)
        result_search = face_search(face_vector)
        st.write(result_search)
        os.remove(image_to_search_path)


    # Display details of page 3
