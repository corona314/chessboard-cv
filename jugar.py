import cv2
import chess
import chess.engine
from vision import VisionTablero

# --- Configuración ---
FUENTE = 2
MOTOR = "stockfish"                    # o ["java", "-jar", "mibot.jar"] si el bot habla UCI
TIEMPO_MOTOR = 0.3
COLOR_ROBOT = chess.WHITE              # chess.WHITE o chess.BLACK: el bando que juega el brazo
VERDE = (0, 255, 0)                    # BGR: origen
ROJO = (0, 0, 255)                     # BGR: destino


def calcular_jugada(engine, board):
    """Solo calcula si es el turno del robot; en el turno humano devuelve None."""
    if board.is_game_over() or board.turn != COLOR_ROBOT:
        return None
    return engine.play(board, chess.engine.Limit(time=TIEMPO_MOTOR)).move


def pedir_jugada(board):
    txt = input("Jugada (UCI ej. e7e5, o SAN ej. e5): ").strip()
    try:
        return board.parse_san(txt)
    except ValueError:
        try:
            return board.parse_uci(txt)
        except ValueError:
            return None


def nombre_color(color):
    return "blancas" if color == chess.WHITE else "negras"


vision = VisionTablero("calibracion.json")
board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci(MOTOR)
# engine.configure({"Skill Level": 5})  # opcional, solo Stockfish
mejor = calcular_jugada(engine, board)   # None si el humano empieza
propuesta = None

cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
if not cap.isOpened():
    raise RuntimeError(f"No se puede abrir la fuente: {FUENTE}")

print(f"Robot juega con {nombre_color(COLOR_ROBOT)}.")
print("ESPACIO=jugada del robot hecha | g=snapshot antes | d=detectar jugada humana | "
      "y=confirmar | m=jugada manual | u=deshacer | x=reiniciar | q/ESC=salir")

mensaje = "Tu turno: pulsa g, haz tu jugada y pulsa d" if mejor is None else ""
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Overlay de la jugada del robot (solo en su turno)
        if mejor is not None:
            vision.pintar(frame, mejor.from_square, VERDE)
            vision.pintar(frame, mejor.to_square, ROJO)
        # Overlay de la jugada humana detectada (pendiente de confirmar)
        if propuesta is not None:
            vision.pintar(frame, propuesta.from_square, (255, 200, 0))
            vision.pintar(frame, propuesta.to_square, (255, 0, 200))

        if board.is_game_over():
            info = f"Fin de partida: {board.result()}"
        elif mejor is not None:
            info = (f"Robot ({nombre_color(board.turn)}): "
                    f"{chess.square_name(mejor.from_square)} -> "
                    f"{chess.square_name(mejor.to_square)}")
        else:
            info = f"Tu turno ({nombre_color(board.turn)})"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, mensaje, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow("Tablero + motor", frame)
        tecla = cv2.waitKey(1) & 0xFF

        if tecla in (ord('q'), 27):
            break

        elif tecla == ord(' ') and mejor is not None:
            captura = board.is_capture(mejor)
            print(f"Robot: {board.san(mejor)} "
                  f"({mejor.uci()}){'  [CAPTURA: retirar pieza antes]' if captura else ''}")
            board.push(mejor)
            mejor = calcular_jugada(engine, board)   # será None: ahora toca al humano
            propuesta = None
            vision.antes = None
            mensaje = "Brazo fuera; pulsa g, haz tu jugada y pulsa d"

        elif tecla == ord('g'):
            vision.guardar_antes(frame)
            mensaje = "Snapshot guardado. Haz tu jugada y pulsa d"

        elif tecla == ord('d'):
            mv, res = vision.detectar(board, frame)
            if mv is None:
                mensaje = f"No detectada: {res}"
                print(mensaje + ". Usa 'm' para introducirla.")
            else:
                propuesta = mv
                mensaje = f"Detectada {board.san(mv)} (conf {res:.0f}). y=confirmar"
                print(mensaje)

        elif tecla == ord('y') and propuesta is not None:
            board.push(propuesta)
            propuesta = None
            vision.antes = None
            mejor = calcular_jugada(engine, board)   # ahora calcula la respuesta del robot
            mensaje = "Jugada aplicada"

        elif tecla == ord('m'):
            mv = pedir_jugada(board)
            if mv is None:
                print("Jugada no válida.")
            else:
                board.push(mv)
                propuesta = None
                vision.antes = None
                mejor = calcular_jugada(engine, board)

        elif tecla == ord('u') and board.move_stack:
            board.pop()
            propuesta = None
            vision.antes = None
            mejor = calcular_jugada(engine, board)

        elif tecla == ord('x'):
            board.reset()
            propuesta = None
            vision.antes = None
            mejor = calcular_jugada(engine, board)
            mensaje = ""
finally:
    engine.quit()
    cap.release()
    cv2.destroyAllWindows()