import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import cv2
import chess
import chess.engine
import pyvirtualcam

from vision import VisionTablero


# =========================
# CONFIGURACIÓN
# =========================

NOMBRE_CAMARA = "wed"      # texto a buscar en el nombre de la cámara
TIEMPO_MOTOR = 0.3
COLOR_ROBOT = chess.WHITE

# True: la jugada detectada se juega directamente al pulsar Enter.
# False: tras detectar hay que pulsar Enter otra vez para confirmar.
AUTOCONFIRMAR = True

# Imagen "estable" = sin movimiento durante este número de fotogramas
FRAMES_ESTABLE = 20
UMBRAL_MOVIMIENTO = 2.0    # diferencia media de gris; sube si nunca se estabiliza

VERDE = (0, 255, 0)
AZUL = (255, 0, 0)
ROJO = (0, 0, 255)
MORADO = (255, 0, 255)

USAR_BOT_JAVA = False

ANCHO = 640
ALTO = 480
FPS = 30

DISPOSITIVO_VIRTUAL = "/dev/video20"


# =========================
# BÚSQUEDA DE CÁMARA POR NOMBRE
# =========================

def buscar_camara(texto):
    """Devuelve N de /dev/videoN cuyo nombre contiene `texto`."""
    texto = texto.lower()
    encontradas = []

    for dev in Path("/sys/class/video4linux").glob("video*"):
        try:
            numero = int(dev.name.replace("video", ""))
            nombre = (dev / "name").read_text().strip()
            indice = int((dev / "index").read_text().strip())
        except (OSError, ValueError):
            continue

        if indice != 0:  # saltar nodos de metadatos
            continue

        if texto in nombre.lower():
            encontradas.append((numero, nombre))

    encontradas.sort()

    if not encontradas:
        raise RuntimeError(
            f"No se encontró ninguna cámara con '{texto}' en el nombre. "
            "Revisa los nombres con: v4l2-ctl --list-devices"
        )

    if len(encontradas) > 1:
        print("Varias cámaras coinciden, se usa la primera:")
        for n, nombre in encontradas:
            print(f"  /dev/video{n}: {nombre}")

    numero, nombre = encontradas[0]
    print(f"Cámara encontrada: /dev/video{numero} ({nombre})")
    return numero


# =========================
# MOTOR DE AJEDREZ
# =========================

if USAR_BOT_JAVA:
    ruta_classpath = (
        Path.home() / "Projects/chessBot/chess-bot/target/classpath.txt"
    )
    classpath = ruta_classpath.read_text(encoding="utf-8").strip()
    MOTOR = ["java", "-cp", classpath, "com.example.UciMain"]
else:
    MOTOR = "stockfish"


def calcular_jugada(engine, board):
    if board.is_game_over() or board.turn != COLOR_ROBOT:
        return None

    resultado = engine.play(board, chess.engine.Limit(time=TIEMPO_MOTOR))
    return resultado.move


def nombre_color(color):
    return "Blancas" if color == chess.WHITE else "Negras"


def mostrar_ayuda():
    print("""
FLUJO NORMAL (solo Enter):
  - Turno del robot: cuando haya movido la pieza, pulsa Enter.
  - Tu turno: mueve tu pieza, quita la mano y pulsa Enter.

Otros comandos:
  n  - Descartar la jugada detectada (si AUTOCONFIRMAR = False)
  m  - Introducir una jugada manualmente (UCI, ej. e2e4)
  u  - Deshacer la última jugada
  x  - Reiniciar la partida
  g  - Guardar a mano la imagen de referencia
  e  - Mostrar el estado
  h  - Ayuda
  q  - Salir
""")


# =========================
# ENTRADA POR TERMINAL
# =========================

comandos = queue.Queue()


def consola():
    while True:
        try:
            cmd = input("> ").strip().lower()

            if cmd in ("", "enter"):
                comandos.put(("enter", None))
            elif cmd in ("m", "manual"):
                jugada = input("Jugada UCI (ej. e2e4): ").strip().lower()
                comandos.put(("m", jugada))
            elif cmd == "q":
                comandos.put(("q", None))
                return
            else:
                comandos.put((cmd, None))

        except (EOFError, KeyboardInterrupt):
            comandos.put(("q", None))
            return


# =========================
# PROGRAMA PRINCIPAL
# =========================

vision = None
engine = None
cap = None
cam_virtual = None

S = SimpleNamespace(
    board=chess.Board(),
    mejor=None,          # jugada sugerida al robot
    propuesta=None,      # jugada detectada (solo si AUTOCONFIRMAR = False)
    pendiente_antes=True,  # guardar referencia en cuanto la imagen sea estable
    estable=False,
    quietos=0,
    limpio=None,
)


def nueva_posicion(guardar_referencia=True):
    """Se llama tras cualquier cambio en el tablero."""
    S.propuesta = None
    vision.antes = None
    S.pendiente_antes = guardar_referencia
    S.mejor = calcular_jugada(engine, S.board)

    if S.board.is_game_over():
        print(f"Partida terminada: {S.board.result()}")
    elif S.mejor is not None:
        print(f"Robot: mueve {S.board.uci(S.mejor)} y pulsa Enter.")
    else:
        print(f"Tu turno ({nombre_color(S.board.turn)}): mueve y pulsa Enter.")


def jugar(mv, etiqueta):
    san = S.board.san(mv)
    S.board.push(mv)
    print(f"{etiqueta}: {san}")
    nueva_posicion()


def detectar_y_jugar():
    if not S.estable:
        print("La imagen aún se mueve. Quita la mano y pulsa Enter otra vez.")
        return

    if getattr(vision, "antes", None) is None:
        print("Aún no hay imagen de referencia. Espera un instante.")
        return

    mv, confianza = vision.detectar(S.board, S.limpio.copy())

    if mv is None:
        print("No se pudo detectar ninguna jugada.")
        return

    if mv not in S.board.legal_moves:
        print(f"Detectada {mv.uci()}, pero no es legal.")
        return

    if confianza is not None:
        print(f"Confianza: {confianza}")

    if AUTOCONFIRMAR:
        jugar(mv, "Jugada detectada")
    else:
        S.propuesta = mv
        print(
            f"Detectada: {S.board.san(mv)} ({mv.uci()}). "
            "Enter para confirmar, n para descartar."
        )


def accion_enter():
    if S.board.is_game_over():
        print(f"Partida terminada: {S.board.result()}")
        return

    # Turno del robot: confirmar que ya ha movido
    if S.board.turn == COLOR_ROBOT:
        if S.mejor is None or S.mejor not in S.board.legal_moves:
            S.mejor = calcular_jugada(engine, S.board)
            if S.mejor is None:
                print("No hay jugada del robot disponible.")
                return
        jugar(S.mejor, "Robot")
        return

    # Turno humano
    if S.propuesta is not None:
        if S.propuesta in S.board.legal_moves:
            jugar(S.propuesta, "Jugada confirmada")
        else:
            S.propuesta = None
            print("La jugada detectada ya no es legal.")
        return

    detectar_y_jugar()


def procesar_comando(cmd, dato):
    """Devuelve True si hay que salir."""
    if cmd == "q":
        return True

    if cmd == "enter":
        accion_enter()

    elif cmd == "n":
        S.propuesta = None
        print("Jugada descartada.")

    elif cmd == "g":
        vision.guardar_antes(S.limpio.copy())
        S.pendiente_antes = False
        print("Imagen de referencia guardada.")

    elif cmd == "m":
        try:
            mv = S.board.parse_uci(dato or "")
        except ValueError:
            print("Jugada no válida. Ejemplo: e2e4")
            return False
        jugar(mv, "Jugada manual")

    elif cmd == "u":
        if not S.board.move_stack:
            print("No hay jugadas que deshacer.")
        else:
            print(f"Jugada deshecha: {S.board.pop().uci()}")
            nueva_posicion(guardar_referencia=False)
            print("Coloca las piezas como estaban y pulsa g.")

    elif cmd == "x":
        S.board.reset()
        print("Partida reiniciada.")
        nueva_posicion(guardar_referencia=False)
        print("Coloca las piezas en la posición inicial y pulsa g.")

    elif cmd == "e":
        print(S.board)
        print(f"Turno: {nombre_color(S.board.turn)}")

    elif cmd == "h":
        mostrar_ayuda()

    else:
        print(f"Comando desconocido: {cmd!r}. Escribe h para ayuda.")

    return False


try:
    FUENTE = buscar_camara(NOMBRE_CAMARA)

    vision = VisionTablero("calibracion.json")

    print("Iniciando motor...")
    engine = chess.engine.SimpleEngine.popen_uci(MOTOR)
    print("Motor iniciado.")

    cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir /dev/video{FUENTE}")

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ANCHO)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ALTO)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    cam_virtual = pyvirtualcam.Camera(
        width=ANCHO,
        height=ALTO,
        fps=FPS,
        device=DISPOSITIVO_VIRTUAL,
        fmt=pyvirtualcam.PixelFormat.BGR,
    )
    print(f"Cámara virtual activa: {cam_virtual.device}")

    mostrar_ayuda()
    threading.Thread(target=consola, daemon=True).start()

    previo = None
    salir = False
    primera = True

    while not salir:
        ok, frame = cap.read()
        if not ok:
            continue

        S.limpio = cv2.resize(frame, (ANCHO, ALTO))

        # --- Detección de estabilidad (sin mano / sin brazo en movimiento) ---
        gris = cv2.cvtColor(
            cv2.resize(S.limpio, (160, 120)), cv2.COLOR_BGR2GRAY
        )
        gris = cv2.GaussianBlur(gris, (5, 5), 0)

        if previo is not None:
            mov = cv2.absdiff(gris, previo).mean()
            S.quietos = S.quietos + 1 if mov < UMBRAL_MOVIMIENTO else 0
        previo = gris
        S.estable = S.quietos >= FRAMES_ESTABLE

        # Posición inicial (necesita ya un fotograma disponible)
        if primera:
            primera = False
            nueva_posicion()

        # Referencia "antes" automática en cuanto la imagen es estable
        if S.pendiente_antes and S.estable:
            vision.guardar_antes(S.limpio.copy())
            S.pendiente_antes = False

        # --- Comandos pendientes ---
        while True:
            try:
                cmd, dato = comandos.get_nowait()
            except queue.Empty:
                break
            try:
                if procesar_comando(cmd, dato):
                    salir = True
                    break
            except Exception as exc:
                print(f"Error: {exc}")

        if salir:
            break

        # --- Salida a la cámara virtual (sin texto) ---
        salida = S.limpio.copy()

        if S.mejor is not None:
            vision.pintar(salida, S.mejor.from_square, ROJO, opacidad=0.35)
            vision.pintar(salida, S.mejor.to_square, VERDE, opacidad=0.35)

        if S.propuesta is not None:
            vision.pintar(salida, S.propuesta.from_square, AZUL, opacidad=0.35)
            vision.pintar(salida, S.propuesta.to_square, MORADO, opacidad=0.35)

        cam_virtual.send(salida)
        cam_virtual.sleep_until_next_frame()


except KeyboardInterrupt:
    print("\nInterrupción recibida. Cerrando...")

finally:
    print("Cerrando recursos...")

    if cam_virtual is not None:
        cam_virtual.close()
    if cap is not None:
        cap.release()
    if engine is not None:
        try:
            engine.quit()
        except Exception:
            pass

    print("Programa finalizado.")