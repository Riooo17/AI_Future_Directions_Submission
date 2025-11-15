# deployment_stub.py
def deploy_to_pi_instructions():
    instructions = """
    Deployment steps:
    1. Copy models/recycle_model.tflite to the Raspberry Pi (scp or rsync).
    2. On Pi: install dependencies (prefer tflite-runtime for Pi):
       sudo apt update
       sudo apt install python3-pip
       pip3 install --upgrade pip
       pip3 install opencv-python-headless numpy
       pip3 install tflite-runtime
    3. Run: python3 pi/pi_infer.py
    4. Measure latency and log outputs for your report.
    """
    print(instructions)

if __name__ == '__main__':
    deploy_to_pi_instructions()
