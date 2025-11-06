Great! Let’s continue writing the README from where we left off ✅

---

# 📘 README (Continue) – Installation & Usage Guide

---

## ⚙️ **1. Installation & Setup**

### ✅ **Step 1: Clone the Project**

```bash
git clone <your-repo-url>
cd Secure_Face_Project
```

### ✅ **Step 2: Create Virtual Environment**

```bash
python -m venv openvino_secure_env
openvino_secure_env\Scripts\activate   # Windows
```

### ✅ **Step 3: Install Dependencies**

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**

```
flask
mtcnn
keras-facenet
opencv-python
numpy
tensorflow
```

---

## 🎬 **2. How to Run the Web App**

### ✅ Start the Server

```bash
python app.py
```

### ✅ Open in Browser

```
http://127.0.0.1:5000/
```

---

## 🖥️ **3. How to Use the Web Application**

✅ **Step 1: Upload Video (.mp4, .avi, .mov, .mkv)**
✅ **Step 2: Upload 1 or more face images (jpg/png/jpeg)**
✅ **Step 3: Set parameters:**

| Field     | Meaning                                          |
| --------- | ------------------------------------------------ |
| Threshold | How *strictly* to match faces (0.4 – 0.6 = good) |
| Blur Type | Gaussian blur / Pixel blur                       |

✅ **Step 4: Click “Start Processing”**
✅ **Step 5: Processing page will show progress (%)**
✅ **Step 6: When done — download the blurred video 🎉**

---

## 🧪 **4. Role of FaceNet & MTCNN in Your App**

| Task                             | Model                       | Purpose                                            |
| -------------------------------- | --------------------------- | -------------------------------------------------- |
| Detect faces in each frame       | **MTCNN**                   | Finds bounding box of each face                    |
| Generate 512D face embedding     | **FaceNet (keras-facenet)** | Convert face image to numerical identity vector    |
| Compare person with known images | **Cosine Similarity**       | Match person from video with uploaded known images |

---

### ✅ Example Code Snippet – Face Comparison

```python
emb = embedder.embeddings(face_array)[0]    # Face embedding from video
best_score = max(cosine_similarity(emb, ke) for ke in known_embeddings)

if best_score < threshold:
    # Face not matched → blur
else:
    # Face matched → keep visible
```

---

## 📥 **5. Input/Output**

| Input         | Description                            |
| ------------- | -------------------------------------- |
| ✅ Video       | MP4 / AVI etc                          |
| ✅ Face Images | Each face you want to **keep visible** |
| ❌ No Face     | All people will be blurred             |

| Output                       | Description               |
| ---------------------------- | ------------------------- |
| ✅ Blurred Video              | All unknown faces blurred |
| ✅ Downloadable in MP4 format | Yes                       |

---

## 💾 **6. Folder Structure (After Uploads)**

```
uploads/
 ├── videos/
 │     ├── <jobid>_inputvideo.mp4
 ├── faces/
 │     ├── <jobid>_face1.jpg
 │     ├── <jobid>_face2.png
outputs/
 ├── output_<jobid>.mp4
```

---
