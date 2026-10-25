
import json
import cv2
import numpy as np
import chess


# =========================
# CONFIGURACIÓN
# =========================

CELL = 80  # píxeles por casilla en la vista cenital

# Filtros disponibles:
# "ninguno", "clahe", "enfoque", "combinado"
FILTRO_DETECCION = "clahe"

# CLAHE
CLAHE_CLIP_LIMIT = 1.8
CLAHE_GRID_SIZE = (8, 8)

# Enfoque
ENFOQUE_SIGMA = 1.2
ENFOQUE_AMOUNT = 0.5

# Reducir ruido sin eliminar demasiado detalle
KERNEL_BLUR = (5, 5)


class VisionTablero:
    def __init__(self, ruta="calibracion.json"):
        with open(ruta, encoding="utf-8") as f:
            cal = json.load(f)

        self.H = np.array(cal["H"], dtype=np.float64)
        self.rot = cal["rotacion"]
        self.M = self._matriz_warp()

        # Snapshot previo a la jugada humana
        self.antes = None

        # Preprocesamiento configurable
        self.filtro = FILTRO_DETECCION

        # Crear CLAHE una sola vez
        self.clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE,
        )

    # =========================
    # PREPROCESAMIENTO
    # =========================

    def filtrar(self, frame):
        """
        Aplica el filtro elegido antes de analizar la imagen.
        Devuelve una imagen BGR del mismo tamaño.
        """

        if self.filtro == "ninguno":
            return frame.copy()

        if self.filtro == "enfoque":
            blur = cv2.GaussianBlur(
                frame, (0, 0), ENFOQUE_SIGMA
            )
            return cv2.addWeighted(
                frame,
                1.0 + ENFOQUE_AMOUNT,
                blur,
                -ENFOQUE_AMOUNT,
                0,
            )

        if self.filtro in ("clahe", "combinado"):
            # Mejorar luminosidad sin modificar directamente
            # los canales de color.
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            l = self.clahe.apply(l)

            imagen = cv2.cvtColor(
                cv2.merge((l, a, b)),
                cv2.COLOR_LAB2BGR,
            )

            if self.filtro == "clahe":
                return imagen

            blur = cv2.GaussianBlur(
                imagen, (0, 0), ENFOQUE_SIGMA
            )
            return cv2.addWeighted(
                imagen,
                1.0 + ENFOQUE_AMOUNT,
                blur,
                -ENFOQUE_AMOUNT,
                0,
            )

        raise ValueError(
            f"Filtro desconocido: {self.filtro}"
        )

    def configurar_filtro(self, nombre):
        """Permite cambiar el filtro desde el programa."""
        permitidos = {
            "ninguno",
            "clahe",
            "enfoque",
            "combinado",
        }

        if nombre not in permitidos:
            raise ValueError(
                f"Filtro no válido: {nombre}. "
                f"Opciones: {sorted(permitidos)}"
            )

        self.filtro = nombre

        # La referencia anterior debe volver a guardarse
        # tras cambiar el filtro para evitar comparaciones
        # entre imágenes procesadas de forma distinta.
        self.antes = None

        print(f"Filtro de visión: {self.filtro}")
        print("Guarda una nueva referencia antes de detectar.")

    # =========================
    # GEOMETRÍA
    # =========================

    @staticmethod
    def _rotar(pt, k):
        x, y = pt

        for _ in range(k % 4):
            x, y = 8 - y, x

        return (x, y)

    def _matriz_warp(self):
        # Coordenadas de destino (vista cenital) ->
        # coordenadas del tablero sin rotar -> cámara.
        R = np.array(
            [
                [0, -1, 8],
                [1,  0, 0],
                [0,  0, 1],
            ],
            dtype=np.float64,
        )

        Rk = np.linalg.matrix_power(R, self.rot % 4)

        escala = np.diag([
            1.0 / CELL,
            1.0 / CELL,
            1.0,
        ])

        return self.H @ Rk @ escala

    def poligono(self, sq):
        """Devuelve las cuatro esquinas de una casilla."""
        x = chess.square_file(sq)
        y = 7 - chess.square_rank(sq)

        pts = [
            (x, y),
            (x + 1, y),
            (x + 1, y + 1),
            (x, y + 1),
        ]

        pts = np.float32([
            self._rotar(p, self.rot) for p in pts
        ]).reshape(-1, 1, 2)

        return cv2.perspectiveTransform(
            pts, self.H
        ).reshape(-1, 2).astype(np.int32)

    def centro(self, sq):
        """Centro de una casilla en coordenadas de cámara."""
        return self.poligono(sq).mean(axis=0)

    # =========================
    # DIBUJO
    # =========================

    def pintar(self, frame, sq, color, opacidad=0.5):
        poly = self.poligono(sq)
        overlay = frame.copy()

        cv2.fillConvexPoly(overlay, poly, color)

        cv2.addWeighted(
            overlay,
            opacidad,
            frame,
            1 - opacidad,
            0,
            dst=frame,
        )

        cv2.polylines(
            frame,
            [poly],
            True,
            color,
            2,
            cv2.LINE_AA,
        )

    # =========================
    # VISTA CENITAL
    # =========================

    def _vista(self, frame):
        # Filtrar antes de corregir la perspectiva.
        imagen = self.filtrar(frame)

        gris = cv2.cvtColor(
            imagen, cv2.COLOR_BGR2GRAY
        )

        cenital = cv2.warpPerspective(
            gris,
            self.M,
            (8 * CELL, 8 * CELL),
            flags=(
                cv2.WARP_INVERSE_MAP
                | cv2.INTER_LINEAR
            ),
        )

        return cv2.GaussianBlur(
            cenital, KERNEL_BLUR, 0
        )

    # =========================
    # COMPARACIÓN DE CASILLAS
    # =========================

    def guardar_antes(self, frame):
        """Guarda la posición previa a la jugada."""
        self.antes = self._vista(frame)

    def _diferencias(self, despues):
        d = cv2.absdiff(
            self.antes, despues
        ).astype(np.float32)

        # Ignorar bordes de casillas, donde hay
        # más interferencias por perspectiva.
        m = int(CELL * 0.2)

        out = np.zeros(64, dtype=np.float32)

        for sq in chess.SQUARES:
            x0 = chess.square_file(sq) * CELL
            y0 = (7 - chess.square_rank(sq)) * CELL

            region = d[
                y0 + m:y0 + CELL - m,
                x0 + m:x0 + CELL - m,
            ]

            out[sq] = (
                region.mean() if region.size else 0.0
            )

        return out

    # =========================
    # DETECCIÓN DE JUGADA
    # =========================

    def detectar(self, board, frame, margen_min=15.0):
        """
        Devuelve (jugada, confianza) o (None, motivo).

        La confianza es el margen entre las dos mejores
        candidatas, no una probabilidad estadística.
        """

        if self.antes is None:
            return (
                None,
                "falta el snapshot 'antes'; guarda la referencia",
            )

        despues = self._vista(frame)

        diffs = self._diferencias(despues)

        cambios = np.clip(
            diffs - np.median(diffs),
            0,
            None,
        )

        candidatas = []

        for mv in board.legal_moves:
            # Mantener el comportamiento original:
            # promociones únicamente a dama.
            if mv.promotion not in (None, chess.QUEEN):
                continue

            b2 = board.copy()
            b2.push(mv)

            inv = [
                s for s in chess.SQUARES
                if board.piece_at(s) != b2.piece_at(s)
            ]

            resto = [
                s for s in chess.SQUARES
                if s not in inv
            ]

            puntuacion = (
                cambios[inv].sum()
                - cambios[resto].sum()
            )

            candidatas.append((puntuacion, mv))

        if not candidatas:
            return None, "sin jugadas legales"

        candidatas.sort(
            key=lambda t: t[0],
            reverse=True,
        )

        if len(candidatas) == 1:
            return candidatas[0][1], 999.0

        margen = (
            candidatas[0][0]
            - candidatas[1][0]
        )

        if margen < margen_min:
            return None, f"ambigua (margen {margen:.1f})"

        return candidatas[0][1], float(margen)
