import os, uuid, time, threading
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, flash
from werkzeug.utils import secure_filename

# ---------------- Paths ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_VIDEOS = os.path.join(BASE_DIR, "uploads", "videos")
UPLOAD_FACES  = os.path.join(BASE_DIR, "uploads", "faces")
OUTPUTS_DIR   = os.path.join(BASE_DIR, "outputs")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

for p in (UPLOAD_VIDEOS, UPLOAD_FACES, OUTPUTS_DIR, STATIC_DIR):
    os.makedirs(p, exist_ok=True)

ALLOWED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# -------------- App + Models --------------
app = Flask(__name__)
app.secret_key = "change-me"

detector = MTCNN()       # face detector
embedder = FaceNet()     # 512-dim FaceNet embeddings

# -------------- Job registry --------------
JOBS = {}  # job_id -> dict(status, progress (0-100), total, done, output_path, error)

# -------------- Utils --------------
def allowed_ext(filename, exts):
    return os.path.splitext(filename)[1].lower() in exts

def cosine_similarity(a, b):
    a = np.asarray(a); b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def extract_face_embedding_from_image_bgr(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)
    if not faces:
        return None
    # largest face
    x, y, w, h = max(faces, key=lambda f: f['box'][2]*f['box'][3])['box']
    x, y = abs(x), abs(y)
    face = rgb[y:y+h, x:x+w]
    if face.size == 0:
        return None
    face = cv2.resize(face, (160, 160))
    face = np.expand_dims(face, axis=0)
    emb = embedder.embeddings(face)[0]
    return emb

def load_known_embeddings(image_paths):
    embs = []
    for p in image_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        emb = extract_face_embedding_from_image_bgr(img)
        if emb is not None:
            embs.append(emb)
    return embs

def blur_patch(patch, method="gaussian"):
    if method == "pixel":
        # coarse pixelation
        h, w = patch.shape[:2]
        s = max(6, min(h, w) // 12)
        small = cv2.resize(patch, (max(1, w // s), max(1, h // s)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    # default gaussian
    h, w = patch.shape[:2]
    kw = max(15, ((w // 6) | 1))
    kh = max(15, ((h // 6) | 1))
    return cv2.GaussianBlur(patch, (kw, kh), 0)

def process_video_job(job_id, video_path, output_path, known_embeddings, threshold, blur_method):
    try:
        JOBS[job_id].update(status="running", progress=0, error=None)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open input video")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        W  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        JOBS[job_id]["total"] = total

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            for f in faces:
                x, y, w, h = f['box']
                x, y = abs(x), abs(y)
                x2, y2 = x + w, y + h
                x = max(0, x); y = max(0, y); x2 = min(W, x2); y2 = min(H, y2)
                if x2 <= x or y2 <= y:
                    continue

                face_rgb = rgb[y:y2, x:x2]
                if face_rgb.size == 0:
                    continue

                face_resized = cv2.resize(face_rgb, (160, 160))
                face_resized = np.expand_dims(face_resized, axis=0)
                emb = embedder.embeddings(face_resized)[0]

                best = 0.0
                for ke in known_embeddings:
                    s = cosine_similarity(emb, ke)
                    if s > best: best = s

                if best < threshold:
                    patch = frame[y:y2, x:x2]
                    frame[y:y2, x:x2] = blur_patch(patch, blur_method)

                # optional label (comment out to remove)
                color = (0,200,0) if best >= threshold else (0,0,255)
                label = f"{'Allowed' if best>=threshold else 'Unknown'} s={best:.2f}"
                cv2.rectangle(frame, (x,y), (x2,y2), color, 2)
                cv2.putText(frame, label, (x, max(0,y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            writer.write(frame)

            idx += 1
            if total > 0:
                JOBS[job_id]["progress"] = int(idx * 100 / total)
            else:
                # approximate progress if FPS / duration unknown
                JOBS[job_id]["progress"] = min(99, JOBS[job_id]["progress"] + 1)

        writer.release()
        cap.release()

        JOBS[job_id].update(status="done", progress=100, done=True, output_path=output_path)
    except Exception as e:
        JOBS[job_id].update(status="error", error=str(e), done=True, progress=100)

# -------------- Routes --------------
@app.get("/")
def index():
    return render_template("index.html")

@app.post("/process")
def process():
    # Validate video
    video_file = request.files.get("video")
    if not video_file or video_file.filename == "":
        flash("Please attach a video file.")
        return redirect(url_for("index"))
    video_name = secure_filename(video_file.filename)
    if not allowed_ext(video_name, ALLOWED_VIDEO_EXTS):
        flash("Unsupported video format.")
        return redirect(url_for("index"))

    # Save video
    job_id = uuid.uuid4().hex
    vpath = os.path.join(UPLOAD_VIDEOS, f"{job_id}_{video_name}")
    video_file.save(vpath)

    # Faces
    faces = request.files.getlist("faces")
    face_paths = []
    for f in faces:
        if f and f.filename:
            fname = secure_filename(f.filename)
            if allowed_ext(fname, ALLOWED_IMAGE_EXTS):
                fpath = os.path.join(UPLOAD_FACES, f"{job_id}_{fname}")
                f.save(fpath)
                face_paths.append(fpath)

    if not face_paths:
        flash("Upload at least one known face image.")
        return redirect(url_for("index"))

    # Params
    threshold = float(request.form.get("threshold", "0.50"))
    blur_method = request.form.get("blur", "gaussian")  # 'gaussian' or 'pixel'

    # Build embeddings
    known_embeddings = load_known_embeddings(face_paths)
    if not known_embeddings:
        flash("No valid faces found in uploaded images.")
        return redirect(url_for("index"))

    # Prepare output
    out_name = f"output_{job_id}.mp4"
    out_path = os.path.join(OUTPUTS_DIR, out_name)

    # Register job and start background thread
    JOBS[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": 0,
        "done": False,
        "output_path": None,
        "error": None,
        "started": time.time(),
        "filename": out_name
    }

    t = threading.Thread(
        target=process_video_job,
        args=(job_id, vpath, out_path, known_embeddings, threshold, blur_method),
        daemon=True
    )
    t.start()

    # redirect to progress page
    return redirect(url_for("result", job_id=job_id))

@app.get("/result/<job_id>")
def result(job_id):
    job = JOBS.get(job_id)
    if not job:
        flash("Invalid job.")
        return redirect(url_for("index"))
    return render_template("result.html", job_id=job_id, filename=job.get("filename"))

@app.get("/progress/<job_id>")
def progress(job_id):
    job = JOBS.get(job_id, {})
    # minimal info for polling
    return jsonify({
        "status": job.get("status", "unknown"),
        "progress": job.get("progress", 0),
        "done": job.get("done", False),
        "error": job.get("error"),
        "filename": job.get("filename"),
    })

@app.get("/download/<path:filename>")
def download(filename):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=True)
if __name__ == "__main__":
    print("Starting Flask at http://127.0.0.1:5000")
    # change port if needed (e.g., 5050) and set debug=False for production
    app.run(host="127.0.0.1", port=5000, debug=True)
