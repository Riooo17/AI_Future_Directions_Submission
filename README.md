# AI Future Directions — Pioneering Tomorrow’s AI Innovations 🌐🚀

This repository is a complete submission package for the course assignment.

Contents
- REPORT.md — Full written report (essays, methods, ethics)
- notebooks/
  - EdgeAI_Recycle_Prototype.ipynb — training -> TFLite conversion (Colab-ready)
  - SmartAgri_Simulation.ipynb — synthetic IoT simulation + LSTM example
- pi/pi_infer.py — Raspberry Pi / camera TFLite inference script
- deployment/deployment_stub.py — deployment instructions + helper stub
- requirements.txt — Python dependencies
- models/ — placeholder for generated models (.h5, .tflite)
- docs/dataflow.txt — ASCII dataflow diagram

Quick usage (Colab)
1. Upload or mount your dataset in Colab under `/content/dataset`.
2. Open `notebooks/EdgeAI_Recycle_Prototype.ipynb`, set `DATA_DIR` and run training + TFLite conversion.
3. Copy `models/recycle_model.tflite` to your device (Raspberry Pi) and run `python3 pi/pi_infer.py`.

Notes
- On Raspberry Pi: prefer installing `tflite-runtime` rather than full TensorFlow for low memory footprint.
- The notebooks use MobileNetV2 as a starting point and include quantization tips.
