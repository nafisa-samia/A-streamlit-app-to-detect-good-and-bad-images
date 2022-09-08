# Core Pkgs
import streamlit as st
from PIL import Image, ImageEnhance
import os
import easyocr
import cv2
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
import time
from collections import OrderedDict

current_dir = os.path.dirname(__file__)

@st.cache
def load_image(img):
    im = Image.open(img)
    return im


face_cascade = cv2.CascadeClassifier('frecog/haarcascade_frontalface_default.xml')


def detect_faces(our_image):
    new_img = np.array(our_image.convert('RGB'))
    img = cv2.cvtColor(new_img, 1)
    gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return img, faces


def ensure_dir(dir: str) -> str:
    path = os.path.join(current_dir, dir)
    if not os.path.isdir(path):
        os.mkdir(path)
    return path

def predict(our_image):
    classifier_model = "bg-model.h5"
    IMAGE_SHAPE = (224, 224,3)
    model = load_model(classifier_model, compile=False, custom_objects={'KerasLayer': hub.KerasLayer})
    our_image = Image.open(our_image)
    test_image = our_image.resize((224,224))
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = [
          'bad',
          'good']
    predictions = model.predict(test_image)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()
    results = {
          'bad': 0,
          'good': 0

}
    result = f"{class_names[np.argmax(scores)]} with a {(100 * np.max(scores)).round(2)} % confidence."
    return result


def load_images_from_dir(folder: str):
    imgs = {}
    for filename in os.listdir(folder):
        print(filename)
        imgs[filename] = Image.open(os.path.join(folder, filename))
    return imgs


def main():
    st.title('Good or Bad Image Detection')
    st.text('ML model to detect whether an image is good or bad')

    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Detection':
        st.subheader("Detection")

        image_files = st.file_uploader("Upload Image", accept_multiple_files=True, type=['jpg', 'png', 'jpeg'])

        our_images = {}
        for image_file in image_files:
            if image_file is not None:
                our_image = Image.open(image_file)

                st.text("Original Image")
                # st.write(type(our_image))
                st.image(our_image)
                our_images[image_file.name] = our_image

        print(our_images)

        # Face Detection
        task = ["Faces", "Text", "Prediction"]
        feature_choice = st.sidebar.selectbox("Find Features", task)
        if st.button("Process"):

            if feature_choice == 'Faces':
                for name, our_image in our_images.items():
                    result_img, result_faces = detect_faces(our_image)
                    # st.image(result_img)

                    if len(result_faces) == 0:
                        d = ensure_dir('good')
                        our_image.save(os.path.join(d, name))
                        st.success('No face, Good Image!!!')
                    else:
                        d = ensure_dir('bad')
                        our_image.save(os.path.join(d, name))
                        st.success('Found faces, Bad Image!!!')

            elif feature_choice == 'Text':
                reader = easyocr.Reader(['en'])
                for image_file in image_files:
                    result = reader.readtext(image_file.getvalue())
                    print(result)
                    text = []
                    for (i, j, k) in result:
                        text.append(j)
                    st.write('Recognized text: ' + ' '.join(text))

                    if len(text) < 10:
                        d = ensure_dir('good')
                        our_image.save(os.path.join(d, image_file.name))
                        st.success('Less text, Good Image!!!')
                    else:
                        d = ensure_dir('bad')
                        our_image.save(os.path.join(d, image_file.name))
                        st.success('Lot of texts, Bad Image!!!')

            elif feature_choice == 'Prediction':
                for image_file in image_files:
                    with st.spinner('Model working....'):
                        # plt.imshow(image)
                        plt.axis("off")
                        predictions = predict(image_file)
                        time.sleep(1)
                        st.success('Classified')
                        st.write(predictions)
                        # st.pyplot(fig)




    elif choice == 'About':
        st.subheader("About MM Hackathon Machine Learning POC")
        st.markdown("Built with Streamlit")
        st.text("About us")
        imgs = load_images_from_dir(os.path.join(current_dir, 'about'))
        for name, img in OrderedDict(sorted(imgs.items())).items():
            st.image(img)
            st.caption(name)



if __name__ == '__main__':
    main()