# Pascal VOC Nesne Tespiti (YOLOv8n)

Bu proje, Pascal VOC (2007 + 2012) veri seti üzerinde transfer öğrenme ile eğitilmiş YOLOv8n nesne tespiti modelini ve Gradio tabanlı çalışan web uygulamasını içerir.

## Dosya Yapısı
- `train.ipynb`: Veri seti hazırlığı, model eğitimi ve test kümesi değerlendirme kodları
- `best.pt`: Eğitilmiş model ağırlıkları
- `app.py`: Çalışan Gradio uygulaması
- `requirements.txt`: Gerekli Python kütüphaneleri

## Uygulamayı Çalıştırma
```bash
pip install -r requirements.txt
python app.py
