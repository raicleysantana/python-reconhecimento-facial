import cv2
from simple_facerec import SimpleFacerec
import numpy as np

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model


def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # convert it from BGR to RGB channel and ordering, resize
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # bounding boxes to their respective lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    return (locs, preds)


# Carrega a camera webcam
cap = cv2.VideoCapture(0)

sfr = SimpleFacerec()
sfr.load_encoding_images("imagens/")

# carrega nosso modelo de detector facial serializado do disco
prototxtPath = r"model/deploy.protext"
weightsPath = r"model/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# carrega o modelo do detector de máscara facial do disco
maskNet = load_model("model/mask_detector.model")

while True:
    ret, frame = cap.read()

    if ret:
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        for (box, pred) in zip(locs, preds):
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # desenha caixa delimitadora e texto
            label = "Com Máscara" if mask > withoutMask else "Sem Máscara"
            color = (0, 255, 0) if label == "Com Máscara" else (0, 0, 255)

            # inclui a probabilidade no rótulo
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # exibe o rótulo e o retângulo da caixa delimitadora na saída
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)

        # Detecta faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

            # Plotagem do nome
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

            # Plotagem da area do retangulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)

        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)

        if key == 27:
            break

cap.release()
cv2.destroyAllWindows()
