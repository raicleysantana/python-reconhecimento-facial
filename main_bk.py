import cv2
from simple_facerec import SimpleFacerec

# Carrega a camera webcam
cap = cv2.VideoCapture(0)

sfr = SimpleFacerec()
sfr.load_encoding_images("imagens/")

while True:
    ret, frame = cap.read()

    if ret:
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
