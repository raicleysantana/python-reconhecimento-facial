import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Redimensione o quadro para uma velocidade mais rápida
        self.frame_resizing = 0.4

    def load_encoding_images(self, images_path):
        # Carrega as images do diretório
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} codificando imagens encontradas.".format(len(images_path)))

        # Armazenar codificação e nomes de imagem
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Obtenha o nome do arquivo apenas do caminho do arquivo inicial.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)

            # Obter codificação
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Armazena o nome do arquivo e a codificação do arquivo
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Codificação de imagens carregadas")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Encontre todos os rostos e codificações de rosto no quadro atual do vídeo
        # Converte a imagem da cor BGR (que o OpenCV usa) para a cor RGB (que usa o face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Veja se o rosto é compatível com o(s) rosto(s) conhecido(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Desconhecido"

            # # Se uma correspondência foi encontrada em known_face_encodings, apenas use a primeira.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Ou, em vez disso, use a face conhecida com a menor distância até a nova face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Converta para matriz numpy para ajustar as coordenadas com o redimensionamento do quadro rapidamente
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names
