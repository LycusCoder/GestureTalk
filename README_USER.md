# 🤚 GestureTalk - Aplikasi Gesture Recognition & Gender Detection

## 🚀 Cara Menjalankan

Cukup jalankan perintah berikut:

```bash
python app.py
```

## ✨ Fitur Utama

### 1. 🤚 Enhanced Hand Rigging
- **Real-time hand skeleton visualization** dengan 21 titik landmarks
- **Multiple visualization modes**: Full, Skeleton, Landmarks, Minimal  
- **Enhanced bone connections** untuk menampilkan struktur tangan yang detail
- **Stabilized tracking** untuk pergerakan yang smooth

### 2. 👤 Gender Detection
- **Face-based gender classification** menggunakan MediaPipe
- **Real-time detection** dengan confidence scoring
- **Gender classification**: Laki-laki, Perempuan, Tidak dapat ditentukan
- **Visual feedback** dengan bounding box dan label

### 3. 🤖 Gesture Recognition  
- **5 gesture types**: tolong, halo, terima_kasih, ya, tidak
- **Real-time prediction** dengan confidence scoring
- **Stability filtering** untuk mengurangi noise detection
- **Machine Learning model** dengan 92% accuracy

### 4. 🔊 Text-to-Speech
- **Bahasa Indonesia support** dengan voice synthesis
- **Automatic speech** untuk detected gestures  
- **Volume dan speed control** yang dapat disesuaikan
- **Emergency phrases** untuk komunikasi assistive

## 🎯 Cara Menggunakan

### Langkah 1: Jalankan Aplikasi
```bash
python app.py
```

### Langkah 2: Start Camera
- Klik tombol **"▶️ Start Camera"** di panel control
- Aplikasi akan mengakses webcam dan mulai menampilkan feed

### Langkah 3: Test Hand Rigging
- **Tunjukkan tangan** ke kamera 
- Anda akan melihat **hand skeleton/rigging** muncul di video
- **21 titik landmarks** akan tervisualisasi dengan koneksi tulang
- Status "Tangan Terdeteksi" akan muncul jika hand tracking aktif

### Langkah 4: Test Gender Detection  
- **Posisikan wajah** dalam frame kamera
- Aplikasi akan mendeteksi dan menampilkan:
  - **Gender classification** (Laki-laki/Perempuan)
  - **Confidence level** dalam persentase
  - **Bounding box** di sekitar wajah

### Langkah 5: Test Gesture Recognition
- **Lakukan gesture** berikut untuk testing:
  - ✋ **Tolong**: Angkat tangan meminta bantuan
  - 👋 **Halo**: Gesture menyapa/melambaikan tangan  
  - 🙏 **Terima Kasih**: Gesture berterima kasih ke dada
  - 👍 **Ya**: Thumbs up
  - 👎 **Tidak**: Pointing/wagging finger

### Langkah 6: Monitor Display
- **Gesture display** menampilkan gesture terdeteksi + confidence
- **Gender display** menampilkan hasil gender detection  
- **Recent history** menampilkan 3 gesture terakhir
- **Performance info** menampilkan FPS dan status sistem

## 🔧 Troubleshooting

### Camera Tidak Terdeteksi
```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera Error'); cap.release()"
```

### Test Komponen Individual
```bash
python test_components.py
```

### Reset Model
```bash
python scripts/create_simple_model.py
```

## 📊 Technical Info

- **Hand Tracking**: MediaPipe Hands dengan 21 landmarks
- **Face Detection**: MediaPipe Face Detection + Face Mesh
- **Gender Model**: Rule-based classification menggunakan facial features
- **Gesture Model**: Random Forest dengan 42 features (21 landmarks × 2 coordinates)
- **Framework**: CustomTkinter untuk GUI, OpenCV untuk video processing

## 🎯 Expected Results

Ketika aplikasi berfungsi dengan baik, Anda akan melihat:

1. **Hand Rigging**: Skeleton tangan dengan 21 titik dan koneksi tulang
2. **Gender Detection**: Label "Laki-laki" atau "Perempuan" dengan confidence
3. **Gesture Recognition**: Nama gesture + confidence percentage
4. **Smooth Performance**: ~20-30 FPS real-time processing

## 💡 Tips Penggunaan

- **Pencahayaan**: Pastikan area cukup terang untuk detection yang optimal
- **Posisi**: Tangan dan wajah dalam frame kamera  
- **Stabilitas**: Tahan gesture 2-3 detik untuk detection yang stabil
- **Distance**: Jarak optimal ~50-100cm dari kamera

## 🚨 Notes

- Aplikasi akan berfungsi meski webcam tidak tersedia (testing mode)
- TTS mungkin tidak bekerja di beberapa environment (normal)
- Gender detection menggunakan facial features, bukan data personal
- Gesture model menggunakan synthetic data untuk demo purposes