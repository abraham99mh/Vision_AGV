import cv2
import numpy as np
import matplotlib.pyplot as plt

# Tamaño real del código en metros
tamano_real = 0.1

# Distancia focal de la cámara web en píxeles
distancia_focal = 1000

# Diccionario de ArUco
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
aruco_params = cv2.aruco.DetectorParameters_create()

# Conectar la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer un frame de la cámara web
    ret, frame = cap.read()
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar códigos QR
    qr_detector = cv2.QRCodeDetector()
    qr_data, bbox, _ = qr_detector.detectAndDecode(frame)
    
    # Detectar códigos ArUco
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    # Dibujar los códigos QR y ArUco detectados
    if len(bbox) > 0:
        for i in range(len(bbox)):
            cv2.rectangle(frame, (int(bbox[i][0]), int(bbox[i][1])),
                          (int(bbox[i][0] + bbox[i][2]), int(bbox[i][1] + bbox[i][3])),
                          (0, 255, 0), 2)
    if len(corners) > 0:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
    
    # Calcular la distancia a los códigos QR y ArUco detectados
    if len(bbox) > 0:
        for i in range(len(bbox)):
            tamaño_imagen = max(bbox[i][2], bbox[i][3])
            distancia = (tamano_real * distancia_focal) / tamaño_imagen
            print("Distancia al código QR: {:.2f} metros".format(distancia))
    if len(corners) > 0:
        for i in range(len(corners)):
            tamaño_imagen = np.linalg.norm(corners[i][0][0] - corners[i][0][1])
            distancia = (tamano_real * distancia_focal) / tamaño_imagen
            print("Distancia al código ArUco: {:.2f} metros".format(distancia))
    
    # Mostrar la imagen
    cv2.imshow('frame', frame)
    
    # Salir si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara web y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()