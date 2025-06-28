
import gradio as gr
import numpy as np
import tensorflow as tf
import librosa

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(audio):
    y, sr = librosa.load(audio, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.resize(mfcc, (40, 40)).astype(np.float32)
    mfcc = mfcc.reshape(1, 40, 40, 1)
    interpreter.set_tensor(input_details[0]['index'], mfcc)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])[0]
    label = int(np.argmax(output))
    confidence = float(np.max(output))
    return f"{'Distress' if label == 1 else 'Neutral'} ({confidence:.2f})"

demo = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
    title="Distress Voice Detector"
)

if __name__ == "__main__":
    demo.launch()
