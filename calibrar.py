
import cv2
import json
import numpy as np

FUENTE = 3
ANCHO = 640
ALTO = 480
N_FRAMES = 30

PATTERN = (7, 7)
CELL = 80

FLAGS = (
    cv2.CALIB_CB_ADAPTIVE_THRESH
    + cv2.CALIB_CB_NORMALIZE_IMAGE
)

CRITERIA = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001,
)

# Coordenadas de los 49 cruces interiores del tablero.
OBJ_PTS = np.array(
    [(c + 1, r + 1) for r in range(7) for c in range(7)],
    dtype=np.float32,
)


def rotar(pt, k):
    x, y = pt
    for _ in range(k % 4):
        x, y = 8 - y, x
    return x, y


def transformar(pt, H, k):
    p = np.float32([[rotar(pt, k)]])
    resultado = cv2.perspectiveTransform(p, H)[0, 0]
    return resultado


cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)

if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir la cámara {FUENTE}")

muestras = []
H = None
rot = 0

print("1. Deja visible el tablero completo y vacío.")
print("2. Cuando aparezca la cuadrícula, pulsa r para cambiar orientación.")
print("3. Pulsa s para guardar y q para salir.")


try:
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        # MISMA resolución que utiliza el programa principal.
        frame = cv2.resize(frame, (ANCHO, ALTO))
        vista = frame.copy()

        if H is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            found, corners = cv2.findChessboardCorners(
                gray, PATTERN, flags=FLAGS
            )

            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (5, 5), (-1, -1), CRITERIA
                )

                muestras.append(corners.reshape(-1, 2))
                cv2.drawChessboardCorners(
                    vista, PATTERN, corners, found
                )

            cv2.putText(
                vista,
                f"Muestras: {len(muestras)}/{N_FRAMES}",
                (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
            )

            if len(muestras) >= N_FRAMES:
                media = np.median(
                    np.asarray(muestras), axis=0
                ).astype(np.float32)

                H, inliers = cv2.findHomography(
                    OBJ_PTS, media, method=0
                )

                if H is None or not np.isfinite(H).all():
                    H = None
                    muestras.clear()
                    print("Homografía inválida. Reintentando.")
                else:
                    print("Calibración calculada.")

        else:
            # Dibujar las líneas de la cuadrícula proyectadas.
            for i in range(9):
                p1 = transformar((i, 0), H, rot)
                p2 = transformar((i, 8), H, rot)
                p3 = transformar((0, i), H, rot)
                p4 = transformar((8, i), H, rot)

                cv2.line(
                    vista, tuple(np.rint(p1).astype(int)),
                    tuple(np.rint(p2).astype(int)),
                    (0, 255, 255), 1, cv2.LINE_AA
                )
                cv2.line(
                    vista, tuple(np.rint(p3).astype(int)),
                    tuple(np.rint(p4).astype(int)),
                    (0, 255, 255), 1, cv2.LINE_AA
                )

            # Etiquetas de orientación.
            for nombre, (f, r) in {
                "a1": (0, 0),
                "h1": (7, 0),
                "a8": (0, 7),
                "h8": (7, 7),
            }.items():
                p = transformar((f + 0.5, 7 - r + 0.5), H, rot)
                cv2.putText(
                    vista, nombre,
                    tuple(np.rint(p).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 255), 1, cv2.LINE_AA
                )

            cv2.putText(
                vista, f"Rotacion: {rot} | r: rotar  s: guardar  q: salir",
                (5, 15), cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (255, 255, 255), 1
            )

        cv2.imshow("Calibracion", vista)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord("q"), 27):
            break

        if H is not None and tecla == ord("r"):
            rot = (rot + 1) % 4

        if H is not None and tecla == ord("s"):
            datos = {
                "H": H.tolist(),
                "rotacion": rot,
                "resolucion": [ANCHO, ALTO],
            }

            with open("calibracion.json", "w") as archivo:
                json.dump(datos, archivo, indent=2)

            print("Guardado calibracion.json")
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
