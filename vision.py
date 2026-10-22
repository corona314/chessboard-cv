import json
import cv2
import numpy as np
import chess

CELL = 80  # píxeles por casilla en la vista cenital


class VisionTablero:
    def __init__(self, ruta="calibracion.json"):
        with open(ruta) as f:
            cal = json.load(f)
        self.H = np.array(cal["H"], dtype=np.float64)
        self.rot = cal["rotacion"]
        self.M = self._matriz_warp()
        self.antes = None  # snapshot previo a la jugada humana

    # ---------- geometría ----------
    @staticmethod
    def _rotar(pt, k):
        x, y = pt
        for _ in range(k % 4):
            x, y = 8 - y, x
        return (x, y)

    def _matriz_warp(self):
        R = np.array([[0, -1, 8], [1, 0, 0], [0, 0, 1]], dtype=float)
        Rk = np.linalg.matrix_power(R, self.rot % 4)
        return self.H @ Rk @ np.diag([1 / CELL, 1 / CELL, 1])

    def poligono(self, sq):
        """4 esquinas de la casilla en píxeles de la cámara."""
        x = chess.square_file(sq)
        y = 7 - chess.square_rank(sq)
        pts = [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)]
        pts = np.float32([self._rotar(p, self.rot) for p in pts]).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2).astype(np.int32)

    def centro(self, sq):
        """Píxel del centro de la casilla (útil luego para el brazo)."""
        return self.poligono(sq).mean(axis=0)

    # ---------- dibujo ----------
    def pintar(self, frame, sq, color, opacidad=0.5):
        poly = self.poligono(sq)
        overlay = frame.copy()
        cv2.fillConvexPoly(overlay, poly, color)
        cv2.addWeighted(overlay, opacidad, frame, 1 - opacidad, 0, dst=frame)
        cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)

    # ---------- detección de jugada ----------
    def _vista(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.warpPerspective(g, self.M, (8 * CELL, 8 * CELL),
                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
        return cv2.GaussianBlur(g, (5, 5), 0)

    def guardar_antes(self, frame):
        self.antes = self._vista(frame)

    def _diferencias(self, despues):
        d = cv2.absdiff(self.antes, despues).astype(np.float32)
        m = int(CELL * 0.2)
        out = np.zeros(64, dtype=np.float32)
        for sq in chess.SQUARES:
            x0 = chess.square_file(sq) * CELL
            y0 = (7 - chess.square_rank(sq)) * CELL
            out[sq] = d[y0 + m:y0 + CELL - m, x0 + m:x0 + CELL - m].mean()
        return out

    def detectar(self, board, frame, margen_min=15.0):
        """Devuelve (jugada, confianza) o (None, motivo)."""
        if self.antes is None:
            return None, "falta el snapshot 'antes' (tecla g)"

        diffs = self._diferencias(self._vista(frame))
        cambios = np.clip(diffs - np.median(diffs), 0, None)

        candidatas = []
        for mv in board.legal_moves:
            if mv.promotion not in (None, chess.QUEEN):
                continue  # promociones siempre a dama
            b2 = board.copy()
            b2.push(mv)
            inv = [s for s in chess.SQUARES if board.piece_at(s) != b2.piece_at(s)]
            resto = [s for s in chess.SQUARES if s not in inv]
            candidatas.append((cambios[inv].sum() - cambios[resto].sum(), mv))

        if not candidatas:
            return None, "sin jugadas legales"
        candidatas.sort(key=lambda t: t[0], reverse=True)
        if len(candidatas) == 1:
            return candidatas[0][1], 999.0

        margen = candidatas[0][0] - candidatas[1][0]
        if margen < margen_min:
            return None, f"ambigua (margen {margen:.1f})"
        return candidatas[0][1], margen