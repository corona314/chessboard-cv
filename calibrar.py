import cv2
import json
import numpy as np

FUENTE = 4 # 0 o 2
N_FRAMES = 30
pattern = (7, 7)
flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj_pts = np.array([(c + 1, r + 1) for r in range(7) for c in range(7)], np.float32)


def rotar(pt, k):
    x, y = pt
    for _ in range(k % 4):
        x, y = 8 - y, x
    return (x, y)


def punto_casilla(file, rank, k, H, dx=0.5, dy=0.5):
    """Píxel de un punto dentro de la casilla (file 0-7 = a-h, rank 0-7 = 1-8)."""
    p = rotar((file + dx, 7 - rank + dy), k)
    return cv2.perspectiveTransform(np.float32([[p]]), H)[0, 0]


cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

muestras, H, rot = [], None, 0
print("Fase 1: tablero vacío y visible. Fase 2: 'r' rota hasta que a1 esté bien, 's' guarda.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if H is None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            muestras.append(corners.reshape(-1, 2))
            cv2.drawChessboardCorners(frame, pattern, corners, found)
        cv2.putText(frame, f"Capturando {len(muestras)}/{N_FRAMES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if len(muestras) >= N_FRAMES:
            media = np.median(np.array(muestras), axis=0)   # mediana: robusta a ruido
            H, _ = cv2.findHomography(obj_pts, media)
    else:
        for f in range(8):
            for r in range(8):
                c = tuple(int(v) for v in punto_casilla(f, r, rot, H))
                cv2.circle(frame, c, 3, (0, 255, 255), -1)
        for nombre, (f, r) in {"a1": (0, 0), "h1": (7, 0), "a8": (0, 7)}.items():
            c = tuple(int(v) for v in punto_casilla(f, r, rot, H))
            cv2.putText(frame, nombre, c, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "r=rotar  s=guardar  q=salir", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Calibracion", frame)
    t = cv2.waitKey(1) & 0xFF
    if t in (ord('q'), 27):
        break
    if H is not None and t == ord('r'):
        rot = (rot + 1) % 4
    if H is not None and t == ord('s'):
        centros = {}
        for f in range(8):
            for r in range(8):
                nombre = "abcdefgh"[f] + str(r + 1)
                centros[nombre] = punto_casilla(f, r, rot, H).tolist()
        with open("calibracion.json", "w") as fh:
            json.dump({"H": H.tolist(), "rotacion": rot,
                       "resolucion": [frame.shape[1], frame.shape[0]],
                       "centros_px": centros}, fh, indent=2)
        print("Guardado en calibracion.json")
        break

cap.release()
cv2.destroyAllWindows()