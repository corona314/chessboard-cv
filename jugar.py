import cv2
import numpy as np
import chess
import chess.engine
import json

# --- Configuración ---
FUENTE = 2
STOCKFISH = "stockfish"
TIEMPO_MOTOR = 0.3
OPACIDAD = 0.5
VERDE = (0, 255, 0)
ROJO = (0, 0, 255)


def poligono_casilla(sq, rotacion, H):
    """Devuelve las 4 esquinas de la casilla en píxeles."""
    x = chess.square_file(sq)
    y = 7 - chess.square_rank(sq)

    pts = [
        (x, y),
        (x + 1, y),
        (x + 1, y + 1),
        (x, y + 1)
    ]

    for _ in range(rotacion % 4):
        pts = [(8 - y, x) for x, y in pts]

    pts = np.float32(pts).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2).astype(np.int32)


def pintar_casilla(frame, poly, color):
    overlay = frame.copy()
    cv2.fillConvexPoly(overlay, poly, color)
    cv2.addWeighted(
        overlay, OPACIDAD, frame, 1 - OPACIDAD, 0, dst=frame
    )
    cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)


def calcular_jugada(engine, board):
    if board.is_game_over():
        return None
    return engine.play(
        board, chess.engine.Limit(time=TIEMPO_MOTOR)
    ).move


# --- Cargar calibración guardada ---
with open("calibracion.json", "r", encoding="utf-8") as f:
    cal = json.load(f)

H = np.array(cal["H"], dtype=np.float64)
rotacion = int(cal["rotacion"])


# --- Inicio de partida ---
board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH)
mejor = calcular_jugada(engine, board)

cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    engine.quit()
    raise RuntimeError(f"No se puede abrir la fuente: {FUENTE}")

print(
    "Teclas: ESPACIO=aplicar jugada sugerida | "
    "m=introducir jugada rival | u=deshacer | "
    "x=reiniciar | q/ESC=salir"
)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        estado = "Calibracion cargada desde JSON"

        if mejor is not None:
            pintar_casilla(
                frame,
                poligono_casilla(mejor.from_square, rotacion, H),
                VERDE
            )
            pintar_casilla(
                frame,
                poligono_casilla(mejor.to_square, rotacion, H),
                ROJO
            )

        turno = "Blancas" if board.turn == chess.WHITE else "Negras"

        if mejor is not None:
            info = (
                f"{turno}: "
                f"{chess.square_name(mejor.from_square)} -> "
                f"{chess.square_name(mejor.to_square)}"
            )
        else:
            info = f"Fin de partida: {board.result()}"

        cv2.putText(
            frame, info, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        cv2.putText(
            frame, estado, (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (200, 200, 200), 1, cv2.LINE_AA
        )

        cv2.imshow("Tablero + Stockfish", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord("q"), 27):
            break

        elif tecla == ord(" ") and mejor is not None:
            captura = board.is_capture(mejor)

            print(
                f"Jugada: {board.san(mejor)} "
                f"({chess.square_name(mejor.from_square)} -> "
                f"{chess.square_name(mejor.to_square)})"
                f"{' [CAPTURA: retirar pieza antes]' if captura else ''}"
            )

            board.push(mejor)
            mejor = calcular_jugada(engine, board)

        elif tecla == ord("m"):
            txt = input(
                "Jugada rival (UCI ej. e7e5, o SAN ej. e5): "
            ).strip()

            try:
                try:
                    mv = board.parse_san(txt)
                except ValueError:
                    mv = board.parse_uci(txt)

                board.push(mv)
                mejor = calcular_jugada(engine, board)

            except ValueError:
                print("Jugada no válida.")

        elif tecla == ord("u") and board.move_stack:
            board.pop()
            mejor = calcular_jugada(engine, board)

        elif tecla == ord("x"):
            board.reset()
            mejor = calcular_jugada(engine, board)

finally:
    engine.quit()
    cap.release()
    cv2.destroyAllWindows()