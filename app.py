import time
import gradio as gr
from ultralytics import YOLO

model = YOLO('best.pt')

def detect_objects(image, conf_threshold):
    if image is None:
        return None, "Lütfen bir görsel yükleyin."
    
    t0 = time.time()
    results = model.predict(image, conf=conf_threshold, imgsz=640, verbose=False)[0]
    inf_ms = (time.time() - t0) * 1000

    annotated_img = results.plot()[:, :, ::-1]

    detections = []
    for box in results.boxes:
        cls_name = model.names[int(box.cls[0])]
        conf = float(box.conf[0])
        detections.append(f"- {cls_name}: %{conf*100:.1f}")

    summary = f"Tahmin Süresi: {inf_ms:.1f} ms\n\nTespit Edilenler:\n"
    summary += "\n".join(detections) if detections else "Nesne tespit edilemedi."
    return annotated_img, summary

demo = gr.Interface(
    fn=detect_objects,
    inputs=[
        gr.Image(type="pil", label="Fotoğraf Yükle"),
        gr.Slider(0.1, 0.9, value=0.25, step=0.05, label="Güven Eşiği (Confidence)")
    ],
    outputs=[
        gr.Image(type="numpy", label="Model Çıktısı"),
        gr.Textbox(label="Tespit Özeti ve Süre")
    ],
    title="Pascal VOC Nesne Tespiti (YOLOv8n)",
    description="Bir fotoğraf yükleyin; model 20 Pascal VOC sınıfı içinde nesne tespiti yapsın."
)

if __name__ == "__main__":
    demo.launch()
