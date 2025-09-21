# main.py
from fastapi import FastAPI, UploadFile, Form, HTTPException
from pydantic import BaseModel, HttpUrl
import uvicorn, tempfile, cv2, numpy as np, requests, os
import mediapipe as mp

app = FastAPI(title="Shifu AI – Video Analysis API")

# ---------- Utilidades ----------
def calculate_angle(a, b, c):
    a, b, c = np.array(a, float), np.array(b, float), np.array(c, float)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = abs(radians*180.0/np.pi)
    return 360-angle if angle > 180 else angle

def analyze_video_from_path(path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(path)
    elbow_angles, knee_angles = [], []
    frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frames += 1

        # procesar cada frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Codo derecho (jab)
            RS = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            RE = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            RW = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,    lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
            elbow_angles.append(calculate_angle(RS, RE, RW))

            # Rodilla derecha (patada)
            RH = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x,   lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            RK = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,  lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            RA = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            knee_angles.append(calculate_angle(RH, RK, RA))

    cap.release()

    # KPIs simples
    def pct_over(arr, thr):
        arr = np.array(arr, float)
        return float(100.0 * np.mean(arr >= thr)) if arr.size else 0.0

    fps = cv2.VideoCapture(path).get(cv2.CAP_PROP_FPS) or 30
    ang = np.array(elbow_angles, float)
    if ang.size == 0: ang = np.array([np.nan])
    # velocidad angular (grados/s)
    v = np.diff(np.nan_to_num(ang, nan=np.nanmean(ang))) * fps
    vmean = float(np.mean(np.abs(v))) if v.size else 0.0
    vmax  = float(np.max(np.abs(v))) if v.size else 0.0

    pct_ext_jab = pct_over(elbow_angles, 160)     # % de frames con extensión >= 160°
    pct_ext_knee = pct_over(knee_angles, 170)     # % de frames con extensión >= 170°
    jab_reps = int(np.sum(np.array(elbow_angles) >= 160))  # heurística simple

    # Score combinado 0-100
    score = 0.5 * pct_ext_jab + 0.5 * pct_ext_knee

    return {
        "frames": int(frames),
        "fps": float(fps),
        "jab_reps": jab_reps,
        "pct_ext_jab": float(pct_ext_jab),
        "pct_ext_knee": float(pct_ext_knee),
        "velocidad_media": float(vmean),
        "vmax": float(vmax),
        "score": float(score),
        "elbow_angles": elbow_angles,  
        "knee_angles": knee_angles  
    }
def process_video_with_overlay(path: str) -> str:
    """Procesa el video y devuelve la ruta de un archivo temporal con landmarks dibujados"""
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    out = cv2.VideoWriter(tmp_out.name, fourcc, cap.get(cv2.CAP_PROP_FPS) or 30,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Procesar pose
        results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        out.write(frame)

    cap.release()
    out.release()
    return tmp_out.name

import matplotlib.pyplot as plt
def plot_elbow_and_knee(elbow_angles, knee_angles, fps):
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(elbow_angles))/fps, elbow_angles, label="Codo derecho (jab)")
    plt.plot(np.arange(len(knee_angles))/fps, knee_angles, label="Rodilla derecha (patada)")
    plt.axhline(160, color="red", linestyle="--", label="Umbral jab")
    plt.axhline(170, color="green", linestyle="--", label="Umbral patada")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (°)")
    plt.title("Evolución de ángulos durante el video")
    plt.legend()

    tmp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp_plot.name)
    plt.close()
    return tmp_plot.name


def plot_elbow(elbow_angles, fps):
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(elbow_angles))/fps, elbow_angles, label="Codo (jab)")
    plt.axhline(160, color="red", linestyle="--", label="Umbral jab")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (°)")
    plt.title("Evolución de Jabs (codo derecho)")
    plt.legend()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name); plt.close()
    return tmp.name

def plot_knee(knee_angles, fps):
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(len(knee_angles))/fps, knee_angles, label="Rodilla (patada)")
    plt.axhline(170, color="green", linestyle="--", label="Umbral patada")
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Ángulo (°)")
    plt.title("Evolución de Patadas (rodilla derecha)")
    plt.legend()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    plt.savefig(tmp.name); plt.close()
    return tmp.name

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

def create_pdf_report(res, plots, filename="report.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    # Portada
    story.append(Paragraph(f"<b>Reporte de Análisis – Shifu AI</b>", styles['Title']))
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"Nombre: {res['nombre']}", styles['Normal']))
    story.append(Paragraph(f"Email: {res['email']}", styles['Normal']))
    story.append(Paragraph(f"Disciplina: {res['disciplina']}", styles['Normal']))
    story.append(Paragraph(f"Nivel: {res['nivel']}", styles['Normal']))
    story.append(Spacer(1, 20))

    # KPIs
    data = [
        ["Frames", res["frames"]],
        ["FPS", round(res["fps"], 2)],
        ["Reps Jab", res["jab_reps"]],
        ["% Extensión Jab", f"{res['pct_ext_jab']:.2f}%"],
        ["% Extensión Rodilla", f"{res['pct_ext_knee']:.2f}%"],
        ["Vel. media", f"{res['velocidad_media']:.2f}°/s"],
        ["Vmax", f"{res['vmax']:.2f}°/s"],
        ["Score", f"{res['score']:.2f}"]
    ]
    story.append(Table(data))
    story.append(Spacer(1, 20))

    # Insertar todas las gráficas con título
    for title, plot in plots:
        story.append(Paragraph(title, styles['Heading2']))
        story.append(Image(plot, width=400, height=200))
        story.append(Spacer(1, 20))


    doc.build(story)
    return filename


def download_to_temp(url: str) -> str:
    # Solo soporta URLs directas a archivo (mp4/mov). Para YouTube necesitarás otro flujo.
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"No se pudo descargar el video: {r.status_code}")
    suffix = ".mp4" if ".mp4" in url.lower() else ".mov"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in r.iter_content(chunk_size=1024*1024):
        if chunk: tmp.write(chunk)
    tmp.close()
    return tmp.name

# ---------- Modelos ----------
class AnalyzeURLPayload(BaseModel):
    nombre: str
    email: str
    disciplina: str
    nivel: str
    video_url: HttpUrl

# ---------- Endpoints ----------
@app.get("/")
def root():
    return {"ok": True, "service": "Shifu AI – Video Analysis API"}




# 1) Subida de archivo (para pruebas locales o Make con multipart/form-data)
@app.post("/analyze_file")
async def analyze_file(
    file: UploadFile,
    nombre: str = Form(...),
    email: str = Form(...),
    disciplina: str = Form(...),
    nivel: str = Form(...)
):
    # guardar temporal
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    tmp.write(await file.read())
    tmp.close()
    res = analyze_video_from_path(tmp.name)
    os.unlink(tmp.name)
    return {"nombre": nombre, "email": email, "disciplina": disciplina, "nivel": nivel, **res}

# 2) URL directa (lo ideal para tu Google Form)
@app.post("/analyze_url")
def analyze_url(p: AnalyzeURLPayload):
    local_path = download_to_temp(str(p.video_url))
    res = analyze_video_from_path(local_path)
    os.unlink(local_path)
    return {"nombre": p.nombre, "email": p.email, "disciplina": p.disciplina, "nivel": p.nivel, **res}


from fastapi.responses import FileResponse
import zipfile

@app.post("/analyze_file_with_video")
async def analyze_file_with_video(
    file: UploadFile,
    nombre: str = Form(...),
    email: str = Form(...),
    disciplina: str = Form(...),
    nivel: str = Form(...)
):
    # Guardar video temporal
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] or ".mp4")
    tmp_in.write(await file.read())
    tmp_in.close()

    # Analizar métricas
    res = analyze_video_from_path(tmp_in.name)
    res.update({"nombre": nombre, "email": email, "disciplina": disciplina, "nivel": nivel})

    # Generar video procesado
    tmp_out = process_video_with_overlay(tmp_in.name)

    # Gráficas
    elbow_plot = plot_elbow(res["elbow_angles"], res["fps"])
    knee_plot = plot_knee(res["knee_angles"], res["fps"])
    both_plot = plot_elbow_and_knee(res["elbow_angles"], res["knee_angles"], res["fps"])
    plots = [
    ("Gráfico de Jabs", elbow_plot),
    ("Gráfico de Patadas", knee_plot),
    ("Gráfico conjunto Jabs + Patadas", both_plot)
    ]

    # Generar PDF
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    create_pdf_report(res, plots, filename=pdf_path)

    # Crear ZIP con video + pdf
    zip_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(tmp_out, f"{nombre}_video.mp4")
        z.write(pdf_path, f"{nombre}_report.pdf")

    # 🚨 Limpieza de archivos temporales
    try:
        os.unlink(tmp_in.name)      # video original
        os.unlink(tmp_out)          # video procesado
        os.unlink(pdf_path)         # pdf
        for _, plot in plots:       # graficas
            os.unlink(plot)
    except Exception as e:
        print("Error limpiando archivos temporales:", e)

    return FileResponse(zip_path, media_type="application/zip", filename="results.zip")
