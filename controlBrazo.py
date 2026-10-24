#!/usr/bin/env python3
"""
Control IDLE / ACTIVO para un brazo SO-101 con una política LeRobot (ACT o SmolVLA).

  IDLE   -> el brazo no ejecuta la política; se queda quieto (o va a la pose de reposo).
  ACTIVO -> la política controla el brazo a ~FPS Hz.

Comandos (escribe y pulsa Enter):
  a  -> pasar a ACTIVO
  i  -> pasar a IDLE (va a la pose de reposo si existe)
  s  -> guardar la pose actual del brazo como pose de reposo (rest_pose.json)
  h  -> ir a la pose de reposo
  q  -> salir (va a reposo antes de desconectar)

Para usarlo desde otro script (p. ej. jugar.py):
    from control_brazo import ControladorBrazo
    ctrl = ControladorBrazo()
    threading.Thread(target=ctrl.run, daemon=True).start()
    ctrl.activar()   # cuando toque mover
    ctrl.reposo()    # cuando no

NOTA: la API de LeRobot cambia entre versiones. Si algún import falla, compáralo
con el ejemplo de SmolVLA/ACT de tu versión instalada (carpeta examples/ del repo).
"""

import json
import threading
import time
from enum import Enum
from pathlib import Path

import torch

from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig

# ----------------------------- CONFIGURACIÓN -----------------------------
POLICY_TYPE = "act"  # "act" o "smolvla"
POLICY_PATH = "outputs/train/act_so101_data15/checkpoints/last/pretrained_model"
TASK = "chess"  # igual que en el dataset

ROBOT_PORT = "/dev/ttyACM0"
ROBOT_ID = "mi_robot"

# Los nombres deben coincidir con los del dataset ("ojos" y "cenital").
# Ajusta los dispositivos: NO pueden ser los mismos que usa otro programa a la vez.
CAMARAS = {
    "ojos": dict(index_or_path="/dev/video1", width=320, height=240),
    "cenital": dict(index_or_path="/dev/video5", width=640, height=480),
}

FPS = 30
MAX_PASO = 15.0  # límite de movimiento por paso (seguridad); None para desactivar
REST_FILE = Path("rest_pose.json")
SEG_HACIA_REPOSO = 2.5  # duración del movimiento suave a reposo
# -------------------------------------------------------------------------


class Estado(Enum):
    IDLE = "IDLE"
    ACTIVO = "ACTIVO"


def cargar_politica(device):
    if POLICY_TYPE == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as Policy
    else:
        from lerobot.policies.act.modeling_act import ACTPolicy as Policy
    policy = Policy.from_pretrained(POLICY_PATH)
    policy.to(device)
    policy.eval()
    return policy


class ControladorBrazo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Dispositivo: {self.device}")

        cams = {n: OpenCVCameraConfig(fps=FPS, **c) for n, c in CAMARAS.items()}
        cfg = SO101FollowerConfig(
            port=ROBOT_PORT, id=ROBOT_ID, cameras=cams, max_relative_target=MAX_PASO
        )
        self.robot = SO101Follower(cfg)
        self.robot.connect()

        self.policy = cargar_politica(self.device)
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=self.policy.config,
            pretrained_path=POLICY_PATH,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )
        self.ds_features = {
            **hw_to_dataset_features(self.robot.action_features, "action"),
            **hw_to_dataset_features(self.robot.observation_features, "observation"),
        }

        self._estado = Estado.IDLE
        self._pedido = None  # "activar" | "reposo" | "ir_reposo" | "guardar_reposo"
        self._lock = threading.Lock()
        self._salir = threading.Event()

    # ------------------------- API pública (thread-safe) -------------------------
    @property
    def estado(self):
        with self._lock:
            return self._estado

    def activar(self):
        self._pedir("activar")

    def reposo(self):
        self._pedir("reposo")

    def ir_reposo(self):
        self._pedir("ir_reposo")

    def guardar_reposo(self):
        self._pedir("guardar_reposo")

    def salir(self):
        self._salir.set()

    def _pedir(self, p):
        with self._lock:
            self._pedido = p

    # ------------------------------ Lógica interna ------------------------------
    def _procesar_pedido(self):
        with self._lock:
            p, self._pedido = self._pedido, None
        if p is None:
            return
        if p == "activar":
            self.policy.reset()  # vacía la cola de acciones anterior
            with self._lock:
                self._estado = Estado.ACTIVO
            print(">> Estado: ACTIVO")
        elif p == "reposo":
            with self._lock:
                self._estado = Estado.IDLE
            self._mover_a_reposo()
            print(">> Estado: IDLE")
        elif p == "ir_reposo":
            self._mover_a_reposo()
        elif p == "guardar_reposo":
            obs = self.robot.get_observation()
            pose = {k: float(v) for k, v in obs.items() if k.endswith(".pos")}
            REST_FILE.write_text(json.dumps(pose, indent=2))
            print(f">> Pose de reposo guardada en {REST_FILE}")

    def _mover_a_reposo(self):
        if not REST_FILE.exists():
            print("(sin pose de reposo guardada; el brazo se queda donde está. Usa 's')")
            return
        destino = json.loads(REST_FILE.read_text())
        obs = self.robot.get_observation()
        inicio = {k: float(obs[k]) for k in destino}
        n = max(1, int(SEG_HACIA_REPOSO * FPS))
        for k in range(1, n + 1):
            a = k / n
            self.robot.send_action({j: inicio[j] + a * (destino[j] - inicio[j]) for j in destino})
            time.sleep(1 / FPS)

    def _paso_politica(self):
        obs = self.robot.get_observation()
        frame = build_inference_frame(
            observation=obs,
            ds_features=self.ds_features,
            device=self.device,
            task=TASK,
            robot_type=self.robot.robot_type,
        )
        frame = self.pre(frame)
        with torch.inference_mode():
            accion = self.policy.select_action(frame)
        accion = self.post(accion)
        accion = make_robot_action(accion, self.ds_features)
        self.robot.send_action(accion)

    def run(self):
        print("Listo. Estado inicial: IDLE")
        try:
            while not self._salir.is_set():
                t0 = time.perf_counter()
                self._procesar_pedido()
                if self.estado == Estado.ACTIVO:
                    self._paso_politica()
                resto = 1 / FPS - (time.perf_counter() - t0)
                if resto > 0:
                    time.sleep(resto)
        finally:
            print("Cerrando...")
            try:
                self._mover_a_reposo()
            finally:
                self.robot.disconnect()


def hilo_teclado(ctrl):
    ayuda = "Comandos: a=activo  i=idle  s=guardar reposo  h=ir a reposo  q=salir"
    print(ayuda)
    while True:
        try:
            c = input("> ").strip().lower()
        except EOFError:
            c = "q"
        if c == "a":
            ctrl.activar()
        elif c == "i":
            ctrl.reposo()
        elif c == "s":
            ctrl.guardar_reposo()
        elif c == "h":
            ctrl.ir_reposo()
        elif c == "q":
            ctrl.salir()
            return
        else:
            print(ayuda)


if __name__ == "__main__":
    ctrl = ControladorBrazo()
    threading.Thread(target=hilo_teclado, args=(ctrl,), daemon=True).start()
    try:
        ctrl.run()
    except KeyboardInterrupt:
        pass