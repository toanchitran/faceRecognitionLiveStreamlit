import streamlit as st
import streamlit.components.v1 as stc

import gc
import os
  
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array, load_img, save_img
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from mtcnn.mtcnn import MTCNN
from PIL import Image
from scipy.spatial.distance import cosine
import pickle

team_pictures = os.path.join("data", "dataset")
face_image = os.path.join("data", "face_image")


def preprocess_image(image_path, required_size=(224, 224)):
    img = load_img(image_path, target_size=required_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


def extract_face(image_path):

    img = cv2.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(img)
    if len(faces) > 0:
        x, y, w, h = faces[0]['box']
        if w > 70:
            detected_face = img[int(y):int(y+h), int(x):int(x+w)]
            return detected_face

    else:
        raise ValueError("Face could not be detected in ", image_path,
                         ". Please confirm that the picture is a face photo.")


model = VGGFace(model='resnet50', include_top=False,
                input_shape=(224, 224, 3), pooling='avg')

def build_representation():
    
    team_members = dict()
    for i in os.listdir(team_pictures):
        total = 0
        team_pics = []
        for j in os.listdir(os.path.join(team_pictures, i)):
            if ('.jpg' in j.lower()) or ('.png' in j.lower()):
                try:
                    exact_path = os.path.join(team_pictures, i, j)
                    detected_face = extract_face(image_path=exact_path)
                    if not os.path.exists(os.path.join(face_image, i)):
                        os.makedirs(os.path.join(face_image, i))
                    if len(os.listdir(os.path.join(face_image, i))) > 0:
                        total = len(os.listdir(os.path.join(face_image, i)))
                    mem_img_path = os.path.sep.join([face_image, i, '%s.png' % (str(total).zfill(5))])
                    cv2.imwrite(mem_img_path, detected_face)
                    total += 1
                    team_image = preprocess_image(mem_img_path)
                    team_pic = model.predict(team_image)[0, :]
                    team_pics.append(team_pic)
                except:
                    pass
        team_members[i] = team_pics

    f = open(os.path.join(team_pictures, "representations.pkl"), "wb")
    pickle.dump(team_members, f)
    f.close()

st.markdown("# FACE RECOGNITION LIVE")
st.markdown("### Build and load representation image of team members")
file_name = os.path.join(team_pictures, "representations.pkl")
if os.path.isfile(file_name):
    st.warning("WARNING: Representations for team images were previously stored in %s. If you added new instances after this file creation, then please delete this file and call find function again. It will create it again."%(file_name))
    f = open(file_name, 'rb')
    team_members = pickle.load(f)
    st.success("Team member representations retrieved successfully")
    st.markdown("Here are the list of all members in team")
    st.write(team_members.keys())
else:
    st.info("Start building the representation file for your team members")

    build_representation()
    f = open(file_name, 'rb')
    team_members = pickle.load(f)
    st.success("Team member representations retrieved successfully")
    st.markdown("Here are the list of all members in team")
    st.write(team_members.keys())

if st.checkbox("Do you want to add new member face?", key='checkbox1'):
    st.markdown("Please give us you member name")
    name = st.text_input("Your member name")
    st.info("Press Button to take the photo")
    if not os.path.exists(os.path.join(team_pictures, name)):
        os.makedirs(os.path.join(team_pictures, name))
    total = len(os.listdir(os.path.join(team_pictures, name)))
    save_path = os.path.join(team_pictures, name)
    run_2 = st.checkbox('Run', key='checkbox3')
    FRAME_WINDOW = st.image([])
    video_file = cv2.VideoCapture(0)
    while run_2:
        
        _, frame = video_file.read()
        frame_2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_2)
        
        col1, col2, col3= st.beta_columns(3)
            
        with col2:
            if st.button("Take photo", key='buttonphoto3'):
                p = os.path.sep.join([save_path, "{}.png".format(str(total).zfill(5))])
                cv2.imwrite(p,frame)
                total += 1

            

    if st.checkbox("Do you finish?", key='checkbox5'):
        st.info("Start add new member image to represenation database")
        team_pics = []
        for j in os.listdir(os.path.join(team_pictures, name)):
            if ('.jpg' in j.lower()) or ('.png' in j.lower()):
                try:
                    exact_path = os.path.join(team_pictures, name, j)
                    detected_face = extract_face(image_path=exact_path)
                    if not os.path.exists(os.path.join(face_image, name)):
                        os.makedirs(os.path.join(face_image, name))
                    mem_img_path = os.path.sep.join([face_image, name, '%s.png' % (str(total).zfill(5))])
                    cv2.imwrite(mem_img_path, detected_face)
                    total += 1
                    team_image = preprocess_image(mem_img_path)
                    team_pic = model.predict(team_image)[0, :]
                    team_pics.append(team_pic)
                except:
                    pass
        team_members[name] = team_pics
        st.success("Team member representations was built successfully")
        st.markdown("Here are the list of all members in team")
        st.write(team_members.keys())
        st.info("Start saving to the representations.pkl file at %s"%(file_name))
        f = open(os.path.join(team_pictures, "representations.pkl"), "wb")
        pickle.dump(team_members, f)
        f.close()
        st.success("Successfully save to representations.pkl file")





# ####################
st.markdown("## Face Recoginition ")
WEIGHT = 'yolo\yolov3-wider_16000.weights'
MODEL = 'yolo\yolov3-face.cfg'

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


run = st.checkbox("Run", key='checkbox2')
FRAME_WINDOW = st.image([])
video_capture = cv2.VideoCapture(0)


while run:
    
    ret, frame = video_capture.read()
    IMG_WIDTH, IMG_HEIGHT = 416, 416

    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(frame, 1/255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction'
    outs = net.forward(output_layers)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

# Scan through all the bounding boxes output from the network and keep only
# the ones with high confidence scores. Assign the box's class label as the
# class with the highest score.

    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # 1 out has multiple predictions with length of 6
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > CONF_THRESHOLD:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)

                # Find the top left point of the bounding box
                topleft_x = int(center_x - width/2)
                topleft_y = int(center_y - height/2)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non maximum suppression to eliminate redundant
    # overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(
        boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    final_boxes = []
    result = frame.copy()
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        if box[2] > 130:
            final_boxes.append(box)
            topleft_x = box[0]
            topleft_y = box[1]
            width = box[2]
            height = box[3]
                
            try:
                detected_face = result[topleft_y:topleft_y +
                                    height, topleft_x:topleft_x+width]
                detected_face = cv2.resize(
                    detected_face, (224, 224))  # resize to 224x224
                img_pixels = image.img_to_array(detected_face)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels = preprocess_input(img_pixels)
                captured_representation = model.predict(img_pixels)[0, :]
                print(captured_representation)
                found = 0
                for j in [*team_members.keys()]:

                    member_name = j
                    representations = team_members.get(member_name)
                    distance_threshold = 0.3
                    best_distance = 0
                    for representation in representations:
                        distance = cosine(
                            representation, captured_representation)
                        
                        if(distance <= distance_threshold):
                            distance_threshold = distance
                            best_distance = distance
                        
                    if best_distance != 0:
                        text_1 = '%s: with %.2f similarity' % (
                            member_name, 1 - best_distance)
                        # Display the label at the top of the bounding box
                        label_size, base_line = cv2.getTextSize(
                            text_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                        top_2 = max(topleft_y, label_size[1])
                        cv2.putText(result, text_1, (topleft_x, top_2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                    COLOR_RED, 1)

                        found = 1

                        break

                if found == 0:

                    text_1 = 'unknown'
                    # Display the label at the top of the bounding box
                    label_size, base_line = cv2.getTextSize(
                        text_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                    top_2 = max(topleft_y, label_size[1])
                    cv2.putText(result, text_1, (topleft_x, top_2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                COLOR_RED, 1)

                

            except:
                pass

        
            cv2.rectangle(result, (topleft_x, topleft_y),
                        (topleft_x+width, topleft_y+height), (255, 255, 255), 1)

            # Display text about confidence rate above each box

        print('[i] ==> # detected faces: {}'.format(len(final_boxes)))
        print('#' * 60)

        # initialize the set of information we'll displaying on the frame
        info = [
            ('number of faces detected', '{}'.format(len(final_boxes)))
        ]

        for (i, (txt, val)) in enumerate(info):
            text = '{}: {}'.format(txt, val)
            cv2.putText(result, text, (10, (i * 20) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)

    

   

          
                
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(result)
        

