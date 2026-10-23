
import os
import time
import queue
import threading
from pathlib import Path

import cv2
import chess
import chess.engine
import pyvirtualcam

from vision import VisionTablero


# =========================
# CONFIGURACIÓN
# =========================

FUENTE = 2
TIEMPO_MOTOR = 0.3
COLOR_ROBOT = chess.WHITE

VERDE = (0, 255, 0)
AZUL = (255, 0, 0)
ROJO = (0, 0, 255)
MORADO = (255, 0, 255)

USAR_BOT_JAVA = False

ANCHO = 640
ALTO = 480
FPS = 30

DISPOSITIVO_VIRTUAL = "/dev/video0"


# =========================
# MOTOR DE AJEDREZ
# =========================

if USAR_BOT_JAVA:
    ruta_classpath = (
        Path.home()
        / "Projects/chessBot/chess-bot/target/classpath.txt"
    )

    with open(ruta_classpath, "r", encoding="utf-8") as f:
        classpath = f.read().strip()

    MOTOR = [
        "java",
        "-cp",
        classpath,
        "com.example.UciMain",
    ]
else:
    MOTOR = "stockfish"


def calcular_jugada(engine, board):
    """Calcula la jugada del robot cuando le toca jugar."""
    if board.is_game_over():
        return None

    if board.turn != COLOR_ROBOT:
        return None

    resultado = engine.play(
        board,
        chess.engine.Limit(time=TIEMPO_MOTOR),
    )

    return resultado.move


def nombre_color(color):
    return "Blancas" if color == chess.WHITE else "Negras"


def mostrar_estado(board, mejor, propuesta):
    print("\n========== ESTADO ==========")
    print(board)
    print(f"\nTurno: {nombre_color(board.turn)}")

    if board.is_game_over():
        print(f"Partida terminada: {board.result()}")
    else:
        print("Partida en curso.")

    print(
        "Jugada sugerida por el motor:",
        board.uci(mejor) if mejor else "ninguna",
    )

    print(
        "Jugada detectada:",
        board.uci(propuesta) if propuesta else "ninguna",
    )

    print("============================\n")


def mostrar_ayuda():
    print("""
Comandos disponibles:

  g  - Guardar imagen ANTES de realizar la jugada
  d  - Detectar la jugada comparando con la imagen guardada
  y  - Confirmar la jugada detectada
  m  - Introducir manualmente una jugada UCI
  espacio - Confirmar que el robot ha ejecutado su jugada
  u  - Deshacer la última jugada
  x  - Reiniciar la partida
  e  - Mostrar el estado de la partida
  h  - Mostrar esta ayuda
  q  - Salir

Ejemplos de jugadas UCI:
  e2e4
  g1f3
  e7e5
""")


# =========================
# ENTRADA POR TERMINAL
# =========================

comandos = queue.Queue()


def consola():
    """
    Lee los comandos en un hilo independiente.
    No modifica directamente el tablero, el motor ni la visión:
    los comandos se procesan en el hilo principal.
    """
    mostrar_ayuda()

    while True:
        try:
            cmd = input("Comando> ").strip().lower()

            if not cmd:
                continue

            if cmd in ("m", "manual"):
                jugada = input(
                    "Introduce la jugada UCI (por ejemplo e2e4): "
                ).strip().lower()

                comandos.put(("m", jugada))
                continue

            if cmd in ("espacio", "robot"):
                comandos.put(("robot", None))
                continue

            if cmd == "q":
                comandos.put(("q", None))
                return

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

try:
    # Inicializar visión y tablero
    vision = VisionTablero("calibracion.json")
    board = chess.Board()

    # Iniciar motor
    print("Iniciando motor...")
    engine = chess.engine.SimpleEngine.popen_uci(MOTOR)
    print("Motor iniciado.")

    # Abrir cámara física
    print(f"Abriendo cámara física: /dev/video{FUENTE}")

    cap = cv2.VideoCapture(FUENTE, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(
            f"No se pudo abrir la cámara física /dev/video{FUENTE}"
        )

    # Reducir latencia de captura
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG"),
    )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Iniciar cámara virtual
    print(f"Abriendo cámara virtual: {DISPOSITIVO_VIRTUAL}")

    cam_virtual = pyvirtualcam.Camera(
        width=ANCHO,
        height=ALTO,
        fps=FPS,
        device=DISPOSITIVO_VIRTUAL,
        fmt=pyvirtualcam.PixelFormat.BGR,
    )

    print(f"Cámara virtual activa: {cam_virtual.device}")

    # Estado inicial
    mejor = calcular_jugada(engine, board)
    propuesta = None
    salir = False

    # Última imagen original, sin cuadrados ni texto
    limpio = None

    # Iniciar lectura de terminal
    hilo_consola = threading.Thread(
        target=consola,
        daemon=True,
    )
    hilo_consola.start()

    print("\nSistema listo.")
    print("La imagen virtual no incluye texto.")
    print("Escribe 'h' en la terminal para ver los comandos.\n")

    while not salir:
        # Capturar imagen
        ok, frame = cap.read()

        if not ok:
            print("Aviso: no se pudo leer un fotograma.")
            time.sleep(0.05)
            continue

        # Ajustar tamaño y conservar imagen original
        limpio = cv2.resize(frame, (ANCHO, ALTO))
        limpio = limpio.copy()

        # Procesar los comandos pendientes en el hilo principal
        while True:
            try:
                cmd, dato = comandos.get_nowait()
            except queue.Empty:
                break

            if cmd == "q":
                salir = True
                break

            elif cmd == "h":
                mostrar_ayuda()

            elif cmd == "e":
                mostrar_estado(board, mejor, propuesta)

            elif cmd == "g":
                try:
                    vision.guardar_antes(limpio.copy())
                    print("Imagen anterior guardada.")
                except Exception as exc:
                    print(f"Error al guardar la imagen: {exc}")

            elif cmd == "d":
                try:
                    if getattr(vision, "antes", None) is None:
                        print("Primero guarda la imagen con el comando g.")
                        continue

                    mv, confianza = vision.detectar(
                        board,
                        limpio.copy(),
                    )

                    if mv is None:
                        print("No se pudo detectar ninguna jugada.")
                        propuesta = None
                    else:
                        propuesta = mv

                        try:
                            san = board.san(mv)
                        except Exception:
                            san = board.uci(mv)

                        print(
                            f"Jugada detectada: {san} "
                            f"({board.uci(mv)})"
                        )

                        if confianza is not None:
                            print(f"Resultado de detección: {confianza}")

                except Exception as exc:
                    print(f"Error durante la detección: {exc}")

            elif cmd == "y":
                if propuesta is None:
                    print("No hay ninguna jugada detectada para confirmar.")
                    continue

                if propuesta not in board.legal_moves:
                    print("La jugada detectada ya no es legal.")
                    propuesta = None
                    continue

                san = board.san(propuesta)
                board.push(propuesta)

                print(f"Jugada confirmada: {san}")

                propuesta = None
                vision.antes = None
                mejor = calcular_jugada(engine, board)

                if mejor:
                    print(
                        "El motor propone:",
                        board.uci(mejor),
                    )

            elif cmd == "robot":
                if mejor is None:
                    print("No hay ninguna jugada del robot pendiente.")
                    continue

                if mejor not in board.legal_moves:
                    print("La jugada del motor ya no es legal.")
                    mejor = calcular_jugada(engine, board)
                    continue

                san = board.san(mejor)
                board.push(mejor)

                print(f"Jugada del robot confirmada: {san}")

                propuesta = None
                vision.antes = None
                mejor = calcular_jugada(engine, board)

                if mejor:
                    print(
                        "Siguiente jugada del motor:",
                        board.uci(mejor),
                    )

            elif cmd == "m":
                if not dato:
                    print("No se ha introducido ninguna jugada.")
                    continue

                try:
                    mv = board.parse_uci(dato)

                    if mv not in board.legal_moves:
                        print("La jugada no es legal en esta posición.")
                        continue

                    san = board.san(mv)
                    board.push(mv)

                    print(f"Jugada manual registrada: {san}")

                    propuesta = None
                    vision.antes = None
                    mejor = calcular_jugada(engine, board)

                    if mejor:
                        print(
                            "El motor propone:",
                            board.uci(mejor),
                        )

                except ValueError:
                    print(
                        "Formato UCI incorrecto o jugada ilegal. "
                        "Ejemplo válido: e2e4"
                    )

            elif cmd == "u":
                if not board.move_stack:
                    print("No hay jugadas que deshacer.")
                    continue

                deshecha = board.pop()
                print(f"Jugada deshecha: {deshecha.uci()}")

                propuesta = None
                vision.antes = None
                mejor = calcular_jugada(engine, board)

                if mejor:
                    print(
                        "El motor propone:",
                        board.uci(mejor),
                    )

            elif cmd == "x":
                board.reset()
                propuesta = None
                vision.antes = None
                mejor = calcular_jugada(engine, board)

                print("Partida reiniciada.")

                if mejor:
                    print(
                        "El motor propone:",
                        board.uci(mejor),
                    )

            else:
                print(f"Comando desconocido: {cmd!r}. Escribe h para ayuda.")

        if salir:
            break

        # Crear salida visual a partir de la imagen limpia.
        # No se escribe ningún texto sobre esta imagen.

        # Crear imagen limpia de salida, sin texto
        salida = limpio.copy()

        # Jugada sugerida por el motor: verde
        if mejor is not None:
            vision.pintar(salida, mejor.from_square, ROJO, opacidad=0.35)
            vision.pintar(salida, mejor.to_square, VERDE, opacidad=0.35)

        # Jugada detectada por visión: rojo
        if propuesta is not None:
            vision.pintar(salida, propuesta.from_square, AZUL, opacidad=0.35)
            vision.pintar(salida, propuesta.to_square, MORADO, opacidad=0.35)

        # Enviar la imagen con los cuadrados, pero sin texto
        cam_virtual.send(salida)

        # Vista local sin texto
        cv2.imshow("Camara y jugadas", salida)
        cv2.waitKey(1)


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

    cv2.destroyAllWindows()

    print("Programa finalizado.")
