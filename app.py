import os
import streamlit as st
from PIL import Image
import PIL
import uuid
from service.face_compare import main
from service.face_index import face_encoding, face_index
from service.face_search import face_search
import time as T
import cv2
import pickle
import tensorflow as tf
from imutils.video import VideoStream
import imutils
import numpy as np
import sklearn
from service.apploy import compare

# Create a page dropdown
title = st.title("Nhận diện sinh trắc học")
page = st.selectbox("Lựa chọn chức năng", ["Face Compare", "Face Index", "Face Search", "Liveness Detection",
                                           "FingerPrint Matching"])
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
        start_time = T.time()
        distance, result = main(image_face_path, image_idcard_front_path)
        st.write("Kết quả:" + str(result))
        st.write("Tỷ lệ: " + str(1 - distance))
        st.write("Thời gian xử lý: " + str(T.time() - start_time))
        os.remove(image_face_path)
        os.remove(image_idcard_front_path)
elif page == "Face Index":
    upload_face_index = st.file_uploader("Chọn ảnh khuôn mặt index ...")
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
    upload_face_search = st.file_uploader("Chọn ảnh khuôn mặt tìm kiếm ...")
    if upload_face_search is not None:
        image_to_search = Image.open(upload_face_search)
        st.image(image_to_search, caption="Uploaded", use_column_width=True)
        image_to_search_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_to_search.save(image_to_search_path)
    if st.button("Search"):
        start_time = T.time()
        face_vector = face_encoding(image_to_search_path)
        result_search = face_search(face_vector)
        os.remove(image_to_search_path)
        st.write(result_search)
        st.write("Thời gian xử lý: " + str(T.time() - start_time))
elif page == "Liveness Detection":
    proto_path = os.path.sep.join(['face_detector', 'deploy.prototxt'])
    model_path = os.path.sep.join(['face_detector', 'res10_300x300_ssd_iter_140000.caffemodel'])
    detector_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)

    # load the liveness detector model and label encoder from disk
    liveness_model = tf.keras.models.load_model('liveness.model')
    le = pickle.loads(open('label_encoder.pickle', 'rb').read())
    if st.button('Check'):
        vs = VideoStream(src=0).start()
        T.sleep(2)  # wait camera to warmup
        imbox = st.empty()
        while True:
            # grab the frame from the threaded video stream
            # and resize it to have a maximum width of 600 pixels
            frame = vs.read()
            frame = imutils.resize(frame, width=600)

            # grab the frame dimensions and convert it to a blob
            # blob is used to preprocess image to be easy to read for NN
            # basically, it does mean subtraction and scaling
            # (104.0, 177.0, 123.0) is the mean of image in FaceNet
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

            # pass the blob through the network
            # and obtain the detections and predictions
            detector_net.setInput(blob)
            detections = detector_net.forward()

            # iterate over the detections
            for i in range(0, detections.shape[2]):
                # extract the confidence (i.e. probability) associated with the prediction
                confidence = detections[0, 0, i, 2]

                # filter out weak detections
                if confidence > 0.5:
                    # compute the (x,y) coordinates of the bounding box
                    # for the face and extract the face ROI
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype('int')

                    # ensure that the bounding box does not fall outside of the frame
                    startX = max(0, startX)
                    startY = max(0, startY)
                    endX = min(w, endX)
                    endY = min(h, endY)

                    # extract the face ROI and then preprocess it
                    # in the same manner as our training data
                    face = frame[startY:endY, startX:endX]
                    # some error occur here if my face is out of frame and comeback in the frame
                    try:
                        face = cv2.resize(face, (32, 32))
                    except:
                        break
                    face = face.astype('float') / 255.0
                    face = tf.keras.preprocessing.image.img_to_array(face)
                    # tf model require batch of data to feed in
                    # so if we need only one image at a time, we have to add one more dimension
                    # in this case it's the same with [face]
                    face = np.expand_dims(face, axis=0)

                    # pass the face ROI through the trained liveness detection model
                    # to determine if the face is 'real' or 'fake'
                    # predict return 2 value for each example (because in the model we have 2 output classes)
                    # the first value stores the prob of being real, the second value stores the prob of being fake
                    # so argmax will pick the one with highest prob
                    # we care only first output (since we have only 1 input)
                    preds = liveness_model.predict(face)[0]
                    j = np.argmax(preds)
                    label = le.classes_[j]  # get label of predicted class

                    # draw the label and bounding box on the frame
                    label = f'{label}: {preds[j]:.4f}'
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

            # show the output fame and wait for a key press
            # cv2.imshow('Frame', frame)
            imbox.image(frame, channels="BGR")
            key = cv2.waitKey(1) & 0xFF

            if st.button('stop'):
                break
        vs.stop()
elif page == "FingerPrint Matching":
    upload_finger1 = st.file_uploader("Chọn ảnh vân tay 1 ...")
    if upload_finger1 is not None:
        image_finger1 = Image.open(upload_finger1)
        st.image(image_finger1, caption="Uploaded", use_column_width=True)
        image_finger1_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_finger1.save(image_finger1_path)
    upload_finger2 = st.file_uploader("Chọn ảnh vân tay 2 ...")
    if upload_finger2 is not None:
        image_finger2 = Image.open(upload_finger2)
        st.image(image_finger2, caption="Uploaded", use_column_width=True)
        image_finger2_path = "data" + "/" + str(uuid.uuid1()) + ".jpg"
        image_finger2.save(image_finger2_path)
    if st.button('Fingerprint compare'):
        result_fingerprint = compare(image_finger1_path, image_finger2_path)
        st.write(result_fingerprint)