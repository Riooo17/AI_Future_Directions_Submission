# pi_infer.py
# TFLite inference script for Raspberry Pi (camera input)
import cv2
import numpy as np
import time
try:
    import tflite_runtime.interpreter as tflite
except Exception:
    import tensorflow as tf
    tflite = tf.lite

MODEL_PATH = 'models/recycle_model.tflite'  # place model here
INPUT_SIZE = (160,160)

def load_interpreter(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def preprocess(frame):
    img = cv2.resize(frame, INPUT_SIZE)
    x = img.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    x = (x / 127.5) - 1.0  # MobileNetV2 preprocessing
    return x

def main():
    interpreter = load_interpreter(MODEL_PATH)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Cannot open camera')
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        x = preprocess(frame)
        interpreter.set_tensor(input_details[0]['index'], x)
        start = time.time()
        interpreter.invoke()
        latency = (time.time() - start) * 1000
        preds = interpreter.get_tensor(output_details[0]['index'])
        label = int(np.argmax(preds[0]))
        conf = float(np.max(preds[0]))
        cv2.putText(frame, f'label:{label} conf:{conf:.2f} {latency:.1f}ms', (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
        cv2.imshow('TFLite Inference', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
