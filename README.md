<p align="center">
  <img src="docs/logo.png" alt="WaveWay" width="220">
</p>

<h1 align="center">WaveWay</h1>

<p align="center">
  <strong>Sensing de actividad humana con WiFi CSI sobre un ESP32.</strong><br>
  Presencia · respiración · clasificación de actividad — sin cámaras, sin wearables.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey.svg" alt="Platform">
  <img src="https://img.shields.io/badge/PyQt-6.4%2B-41cd52.svg" alt="PyQt6">
  <img src="https://img.shields.io/badge/status-research-orange.svg" alt="Research">
</p>

<p align="center">
  <img src="docs/demo.gif" alt="WaveWay demo" width="720">
</p>

---

## Tabla de contenidos

- [Quick start](#quick-start)
- [Qué es WaveWay](#qué-es-waveway)
- [Capturas](#capturas)
- [Stack](#stack)
- [Estructura del repo](#estructura-del-repo)
- [Instalación](#instalación)
- [Uso](#uso)
- [Cuántos datos hacen falta](#cuántos-datos-hacen-falta)
- [Pipeline técnico](#pipeline-técnico)
- [Limitaciones conocidas](#limitaciones-conocidas)
- [Formato de los pesos](#formato-de-los-pesos)
- [Roadmap](#roadmap)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Quick start

```bash
git clone https://github.com/<usuario>/WaveWay.git
cd WaveWay
pip install -r requirements.txt
python main.py
```

Necesitas un ESP32 emitiendo CSI por **COM6** a **921600 baud** (cambia el puerto en
`ui/app.py` si es otro). Pulsa **Iniciar** → **Calibrar** (con la sala vacía, 5 s) →
deja a alguien moverse delante del enlace ESP32 ↔ router.

---

## Qué es WaveWay

WaveWay extrae **Channel State Information (CSI)** de las subportadoras 802.11n HT20
que viajan entre un router WiFi y un ESP32 receptor. A partir de las amplitudes de
esas subportadoras (48 útiles tras filtrar guardas, pilotos y DC) detecta:

- **Presencia** de personas en el enlace.
- **Respiración** (banda 9–30 rpm) por análisis espectral (Welch + concentración en banda).
- **Actividad** clasificada en 5 categorías: `vacío`, `parado`, `sentado`, `caminando`,
  `tirado` — vía un MLP entrenado con datos propios.
- **Latido** (experimental, 48–150 bpm) cuando el SNR lo permite.

Proyecto de investigación. La plataforma `WaveWay` es la base; cada caso de uso vertical
(`WaveWay Care`, `WaveWay Commerce`, `WaveWay Industrial`…) reutiliza el mismo pipeline
ajustando umbrales, etiquetas y UI.

---

## Capturas

| Ventana principal | Modo entrenamiento |
|---|---|
| ![Main window](docs/screenshot_main.png) | ![Training window](docs/screenshot_training.png) |

---

## Stack

- **Hardware** — 1 ESP32 (802.11n HT20, 2.4 GHz) emitiendo CSI por serial a 921600 baud,
  más un router doméstico como anclaje del enlace.
- **Software** — Python 3.10+, PyQt6, NumPy. Opcional: PyTorch para entrenar más rápido,
  OpenCV + MediaPipe para la cámara de referencia durante la grabación.

---

## Estructura del repo

```
WaveWay/
├── main.py                          # entry point
├── requirements.txt
├── LICENSE
├── README.md
├── docs/                            # logo, screenshots, demo.gif
├── core/
│   ├── csi_reader.py                # serial → frames CSI (amplitud, 48 SCs HT20)
│   ├── detector.py                  # presencia, vitals throttle, posición espectral
│   ├── vitals.py                    # Hampel + Welch PSD + SNR de banda
│   ├── models.py                    # dataclass Detection
│   └── training/
│       ├── activity_classes.py      # catálogo de clases (5)
│       ├── data_collector.py        # acumula muestras (CSI, label, ts)
│       ├── trainer.py               # MLP 256→128→C con cross-entropy ponderada
│       ├── model.py                 # ActivityClassifier (inferencia runtime)
│       └── pose_extractor.py        # MediaPipe (sólo preview de cámara)
├── ui/
│   ├── app.py                       # ventana principal
│   ├── radar_view.py                # plano de la sala (LOS ESP32↔router)
│   ├── activity_view.py             # silueta + clase predicha + confianza
│   ├── vitals_view.py               # gráfica de respiración
│   ├── heatmap_view.py              # waterfall de CSI
│   ├── training_window.py           # grabación + entrenamiento + sesiones
│   └── camera_skeleton_view.py      # webcam con skeleton COCO (preview, no se guarda)
└── training_data/                   # sesiones (.npy) + pesos del modelo (.npz)
```

---

## Instalación

```bash
# Requisitos mínimos (lectura + UI + inferencia)
pip install -r requirements.txt

# Opcionales:
pip install torch                       # entrenamiento acelerado (GPU si disponible)
pip install opencv-python mediapipe     # preview de webcam durante la grabación
```

Sin `torch`, el entrenamiento usa un backend SGD en NumPy (más lento pero funcional).
Sin `mediapipe`, la grabación sigue funcionando — sólo desaparece el esqueleto de cámara.

---

## Uso

### 1. Conexión del ESP32

El firmware del ESP32 (basado en el ejemplo CSI de Espressif) imprime una línea
`CSI_DATA,<seq>,<mac>,...,"[I0,Q0,I1,Q1,...]"` por cada frame. El reader espera:

- Puerto serial **COM6** (cambiar en `ui/app.py` si es otro), **921600 baud**.
- Frames a ~100 Hz, **64 subportadoras** (LLTF) o **128** (LLTF+HT-LTF). Se filtran
  guardas, pilotos y DC; quedan **48 subportadoras de datos** HT20.

### 2. Arrancar la app

```bash
python main.py
```

- Pulsa **Iniciar** para abrir el puerto serial y empezar a leer.
- Pulsa **Calibrar** con la sala vacía (5 s) para fijar el umbral de presencia
  y la baseline de atenuación.

### 3. Grabar dataset y entrenar

Botón **Entrenar** → ventana de entrenamiento (3 pestañas):

1. **Grabación**
   - Elige la clase en el dropdown (`vacío`, `parado`, …).
   - Cambia el nombre de sesión (se auto-rellena con `<clase>_001`).
   - Pulsa **Grabar**, mantén la pose 60–90 s, **Detener**.
   - Cada sesión guarda 3 archivos: `<name>_csi.npy`, `<name>_labels.npy`, `<name>_ts.npy`.

2. **Entrenamiento**
   - Muestra muestras por clase. Pulsa **Iniciar entrenamiento**.
   - 60 épocas con z-score por subportadora + cross-entropy ponderada por frecuencia
     de clase. Genera `training_data/activity_model.npz`.
   - Mira `val_acc` en los logs: > 0.80 listo; 0.60–0.80 demo aceptable; < 0.60 graba más.

3. **Muestras**
   - Lista todas las sesiones del disco, con renombrar/eliminar.

Al cerrar la ventana de entrenamiento, la app principal recarga automáticamente el modelo.
A partir de ahí, el panel `ESPECTRO · ACTIVIDAD` muestra la silueta de la clase predicha.

---

## Cuántos datos hacen falta

| Plan | Por clase | Tiempo de grabación | Resultado |
|---|---|---|---|
| Sanity check | 1 × 30 s | ~3 min | Verifica que el pipeline arranca; memoriza. |
| Demo mínima | 2 × 60 s | ~10 min | Funciona en el ambiente exacto de grabación. |
| **Demo decente** | 3–4 × 60–90 s, variando posición | 20–30 min | Recomendado para presentación. |
| Robusto | 6–8 × 90 s con 2 sujetos / 2 zonas | 1–2 h | Generalización aceptable single-node. |

Los datos se guardan a 10 Hz, ventana de 32 muestras (3.2 s). Una sesión de 60 s
genera ~568 ventanas (sliding stride 1). Ver `core/training/trainer.py` para los
hiperparámetros.

---

## Pipeline técnico

### Lectura CSI
- `core/csi_reader.py` abre el puerto, parsea líneas `CSI_DATA`, convierte I/Q a amplitud
  (`sqrt(I² + Q²)`) y aplica una máscara HT20 (drop de DC + guardas + pilotos).
- Buffer circular de 3000 frames (~30 s @ 100 Hz) → resolución FFT ≈ 0.033 Hz para vitals.

### Detección (presencia + vitals)
- `core/detector.py` combina dos vías:
  - **Varianza temporal** entre frames consecutivos (sensible al movimiento off-LOS).
  - **Atenuación amplitud vs. baseline** (sensible al bloqueo del camino directo).
- Calibración fija el umbral en `mean(varianza) + 4·std` sobre los primeros 5 s con sala vacía.
- Vitals (`core/vitals.py`): Hampel para outliers → Welch PSD batched por subportadora
  → fusión por concentración de energía en banda → SNR vs. mediana in-band → confianza.
  Bandas: respiración 0.15–0.50 Hz (9–30 rpm), latido 0.80–2.50 Hz (48–150 bpm, experimental).

### Clasificador de actividad
- MLP `Linear(W·SC) → 256 ReLU → 128 ReLU → C`.
- Entrada: ventana de 32 frames × 48 subportadoras, normalizada con `mu/sigma` por
  subportadora (computados sólo sobre train, guardados en el `.npz`).
- Loss: cross-entropy con pesos inversos a la frecuencia de cada clase (compensa
  desbalance — `vacío` suele dominar en duración).
- Backend: PyTorch (Adam + ExpLR) o fallback NumPy SGD con He-init.
- Inferencia: pura NumPy, sin dependencia de torch en runtime.

---

## Limitaciones conocidas

Lo que un ESP32 único con amplitud-only **no puede** entregar a nivel producto, y por qué:

- **Pose (esqueleto fino tipo COCO)** — requiere ≥3 antenas Rx con fase desenrollada
  (Person-in-WiFi, WiPose). Con 1 Rx amplitud, un MLP memoriza la sesión de
  entrenamiento; no generaliza cross-room.
- **Posición XY centimétrica** — requiere AoA (multi-antena) o ToF (Wi-Fi 6 FTM, UWB).
  El centroide de subportadoras del `radar_view` se mantiene como visualización
  espectral, no como medida espacial.
- **Heart rate** — banda 48–150 bpm con SNR marginal en 2.4 GHz amplitud. Marcado
  como experimental, opt-in con el botón del topbar.

WaveWay sí entrega de forma fiable: **presencia**, **conteo de respiración** (con sujeto
quieto y enlace LOS limpio) y **clasificación gruesa de actividad** (5 clases).

---

## Formato de los pesos

`activity_model.npz` contiene:

```
w1, b1, w2, b2, w3, b3  # MLP weights
num_sc                  # subcarrier count this model was trained on
num_classes             # output dimension
mu, sigma               # per-subcarrier z-score stats from train
class_names             # ordered list of class labels (utf-8 array)
```

---

## Roadmap

- [ ] Canal de fase con linear-fit sanitization (mejora SNR de vitals y abre la puerta
      a AoA grueso con 2 antenas Rx).
- [ ] Multi-ESP32 → fusión por habitación / zona.
- [ ] Detección de caídas como evento (transición `caminando|parado → tirado` + quietud
      prolongada), no como clase estática.
- [ ] Pipeline de validación contra ground truth (webcam + MediaPipe ya está integrado
      como preview; falta exportar etiquetas de validación).
- [ ] Versiones verticales: `WaveWay Care`, `WaveWay Commerce`, `WaveWay Industrial`.

---

## Citation

Si usas WaveWay en una publicación académica, por favor cita:

```bibtex
@software{waveway2026,
  author  = {MilixDev},
  title   = {WaveWay: WiFi CSI-based human activity sensing platform},
  year    = {2026},
  url     = {https://github.com/<usuario>/WaveWay}
}
```

---

## Acknowledgments

- Espressif por el [ejemplo CSI de ESP-IDF](https://github.com/espressif/esp-idf/tree/master/examples/wifi/csi_recv)
  que sirve como firmware del lado ESP32.
- [MediaPipe](https://developers.google.com/mediapipe) por la extracción de pose usada
  en el preview de cámara durante la grabación.
- Trabajos previos en sensing CSI single-link: FallDeFi, WiSpiro, FreeSense, CARM.

---

## License

[MIT](LICENSE) © 2026 MilixDev
