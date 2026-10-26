# chessboard-cv ♟️

Visión por ordenador para un tablero de ajedrez: una cámara mira el tablero,
`jugar.py` lleva la partida y saca la imagen por una **cámara virtual**
(la "cenital"), que luego usa el robot en otra parte del proyecto.

> Aquí **no se ejecuta el robot**. Este repo solo es la parte de la cámara y la partida.

## Qué hace

```
cámara real ──► jugar.py ──► cámara virtual (/dev/video10)
                               └─► cualquier app que lea cámaras (OBS, navegador, LeRobot…)
```

## Qué necesitas

- Linux
- Una cámara USB apuntando al tablero
- Python 3
- `v4l2loopback` (es lo que crea la cámara virtual)

## Instalar

```bash
git clone https://github.com/corona314/chessboard-cv
cd chessboard-cv
python3 -m venv .venv
source .venv/bin/activate
```

## Crear la cámara virtual (una vez)

```bash
sudo apt install v4l2loopback-dkms v4l2loopback-utils
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="ChessCam" exclusive_caps=1
ls -l /dev/video10      # tiene que existir
```

Si ya usas OBS Virtual Camera, el módulo ya está cargado y este comando no hará
nada. En ese caso descárgalo y vuelve a cargarlo con los dos dispositivos:

```bash
sudo modprobe -r v4l2loopback
sudo modprobe v4l2loopback devices=2 video_nr=0,10 card_label="OBS Virtual Camera,ChessCam" exclusive_caps=1,1
```

## Jugar

```bash
source .venv/bin/activate
python jugar.py
```

Mientras corre, abre `/dev/video10` desde cualquier programa (OBS, el navegador,
etc.) y verás el tablero con la partida.

<!-- TODO: opciones de jugar.py, cómo se ponen las piezas, nivel del motor -->

## Si algo falla

| Error | Qué hacer |
|---|---|
| `Device /dev/video10 does not exist` | Falta crear la cámara virtual (mira arriba) |
| La cámara virtual no abre / sale negra | `jugar.py` tiene que estar corriendo primero: la cámara virtual solo funciona mientras alguien le manda imagen |
| `Permission denied` en la cámara real | Añade tu usuario al grupo `video`: `sudo usermod -aG video $USER` y reinicia sesión |
| No sé qué índice es mi cámara | `v4l2-ctl --list-devices` |

## Para el hackathon

La cámara virtual es la vista "cenital" del tablero que usa el robot SO-101
(640×480 a 30 fps). Si quieres conectarlo con LeRobot, apunta tu cámara
`cenital` al dispositivo de la cámara virtual y arranca `jugar.py` **antes**.
