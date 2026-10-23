import cv2
import json, numpy as np

with open("calibracion.json") as f:
    cal = json.load(f)
H = np.array(cal["H"])
rotacion = cal["rotacion"]
centros_px = cal["centros_px"]

FUENTE = 4  # 0 para la webcam, a priori 2 una externa

cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError(f"No se puede abrir la cámara {FUENTE}")
    

# --- Configuración ---

# Tablero de 8x8: 7x7 esquinas interiores
pattern = (7, 7)

flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


while True:
    ok, frame = cap.read()
    if not ok:
        break  # fin del vídeo o error de lectura

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)

    if found:
        # Refinar la posición de las esquinas
        corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
        cv2.drawChessboardCorners(frame, pattern, corners, found)
        texto = f"Tablero detectado: {len(corners)} esquinas"
        color = (0, 255, 0)
    else:
        texto = "No se detecta el tablero"
        color = (0, 0, 255)

    cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2, cv2.LINE_AA)

    cv2.imshow("Deteccion del tablero", frame)

    # Pulsa 'q' o ESC para salir
    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()