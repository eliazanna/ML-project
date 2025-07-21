import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import random
import time
import threading
import base64
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import mediapipe as mp

MODEL_PATH = "D:/DESKTOP/Desktop/ML-project/Model_B/predictor_B.keras"
IMG_W, IMG_H = 300, 200
CLASS_NAMES = ["paper", "rock", "scissors"]
CONF_TH = 0.50

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)
model = load_model()

def predict_move(roi_rgb):    #resize + float32 + normalizzazione
    proc = cv2.resize(roi_rgb, (IMG_W, IMG_H)).astype(np.float32) / 255.0
    probs = model.predict(proc[np.newaxis, ...], verbose=0)[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    return (CLASS_NAMES[idx], conf) if conf >= CONF_TH else ("uncertain", conf)


def decide(u,c): #regole classiche rock, paper, scissors
    if u==c: return "draw"
    return "win" if (u,c) in {("rock","scissors"),("scissors","paper"),("paper","rock")} else "lose"

class FrameBuffer:
    def __init__(self):
        self.lock = threading.Lock()
        self.frame=None
        self.idx=0
frame_buffer = FrameBuffer()

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with frame_buffer.lock:
        frame_buffer.frame = img
        frame_buffer.idx += 1
    return frame

mp_hands = mp.solutions.hands #hand detector
snapshot_hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

#conversione in rgb e rilevamento mano con MediaPipe.
#qiuando na mano è trovata calcola un bounding box espanso, estrae la regione e la restituisce in formato RGB insieme alle coordinate del box.
def detect_and_crop(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = snapshot_hands.process(rgb)
    if not res.multi_hand_landmarks: 
        return None, None, False
    
    h, w, _ = bgr.shape
    lm = res.multi_hand_landmarks[0].landmark
    x_min = int(min(pt.x for pt in lm) * w)
    x_max = int(max(pt.x for pt in lm) * w)
    y_min = int(min(pt.y for pt in lm) * h)
    y_max = int(max(pt.y for pt in lm) * h)

    margin = int(max(x_max - x_min, y_max - y_min) * 0.20) #box rettangolo
    x1, y1 = max(x_min - margin, 0), max(y_min - margin, 0)
    x2, y2 = min(x_max + margin, w), min(y_max + margin, h)

    roi_bgr = bgr[y1:y2, x1:x2]
    if roi_bgr.size == 0:
        return None, None, False

    return cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB), (x1, y1, x2, y2), True


#parte di STREAMLIT
st.set_page_config("RPS", "✊", layout="wide")

st.markdown("""
<style>
body, .stApp {background:#10131a; color:#e6e6e9; font-family:system-ui, sans-serif;}
h1 {text-align:center; font-size:2.1rem; margin:0.4rem 0 0.6rem; letter-spacing:2px;}
.scorebar {display:flex; justify-content:center; gap:1.5rem; margin-bottom:0.5rem;}
.scorecard {background:#1c1f27; padding:0.55rem 1.0rem 0.7rem; border:1px solid #2d323f; border-radius:14px; text-align:center; min-width:105px;}
.scorecard b {font-size:0.6rem; letter-spacing:2px; opacity:0.6;}
.scorecard span {display:block; font-size:1.7rem; line-height:1.4rem; margin-top:0.2rem;}
.layout-row {display:flex; gap:1rem; align-items:flex-start;}
#cam-wrapper {width:260px; position:relative; background:#181b22; border:1px solid #2c3140; border-radius:16px; padding:0.55rem 0.6rem 0.7rem;}
#cam-wrapper video, #cam-wrapper canvas {width:240px !important; height:170px !important; object-fit:cover !important; border-radius:12px; border:2px solid #2c3140;}
#snapshot-layer, #countdown-layer {position:absolute; top:34px; left:10px; width:240px; height:170px; border-radius:12px;}
#snapshot-layer {display:none; overflow:hidden; border:2px solid #2c3140;}
#snapshot-layer img {width:100%; height:100%; object-fit:cover;}
#countdown-layer {display:none; justify-content:center; align-items:center; background:rgba(0,0,0,0.55); font-size:3rem; font-weight:800; letter-spacing:4px; z-index:9;}
#cam-wrapper button[aria-label="Start"], #cam-wrapper button[aria-label="Stop"], #cam-wrapper select {display:none !important;}
.stButton > button {background:#272c39 !important; border:1px solid #384055 !important; color:#f0f0f5 !important; font-size:0.7rem !important; padding:0.5rem 0.65rem !important; border-radius:9px !important; font-weight:600 !important; letter-spacing:1px; width:100%;}
.stButton > button:hover {background:#313847 !important;}
.result-area {flex:1; background:#181b22; border:1px solid #2c3140; border-radius:20px; padding:0.65rem 0.85rem 0.9rem;}
.big-moves {display:flex; gap:0.8rem; flex-wrap:wrap; margin-top:0.4rem;}
.card-move {flex:1 1 220px; background:#1e222b; border:1px solid #2d323f; border-radius:18px; padding:0.75rem 0.6rem 0.95rem; text-align:center; min-height:118px;}
.card-move h2 {margin:0.15rem 0 0.2rem; font-size:1.75rem; letter-spacing:1px;}
.move-label {font-size:0.55rem; text-transform:uppercase; letter-spacing:2px; opacity:0.55;}
.confidence {font-size:0.58rem; opacity:0.5; letter-spacing:1px;}
.banner {text-align:center; margin-top:0.4rem; font-size:2.3rem; font-weight:800; letter-spacing:3px;}
.win {color:#34ff6a;} .lose {color:#ff4a4a;} .draw {color:#ffc93b;}
.sep {height:2px; background:linear-gradient(90deg,rgba(255,255,255,0) 0%,#343a46 50%,rgba(255,255,255,0) 100%); margin:0.4rem auto 0.5rem; width:70%; border-radius:6px;}
</style>
""", unsafe_allow_html=True)

#memoria tasti in streamlit
for k,v in {"score_u":0,"score_c":0,"user_move":"","cpu_move":"","conf":0.0,"result":"","round_running":False,"snapshot_bgr":None,"snapshot_ready":False}.items():
    if k not in st.session_state: st.session_state[k]=v
score_placeholder = st.empty()

def render_score(): #interfaccia
    score_placeholder.markdown(f"""<div class='scorebar'>
        <div class='scorecard'><b>YOU</b><span>{st.session_state.score_u}</span></div>
        <div class='scorecard'><b>CPU</b><span>{st.session_state.score_c}</span></div>
    </div>""", unsafe_allow_html=True)

st.markdown("<h1>ROCK • PAPER • SCISSORS</h1>", unsafe_allow_html=True)
render_score()




#HELPERS 
def capture_next_frame_after(idx_start, timeout=0.4):
    deadline = time.time() + timeout
    while time.time() < deadline:
        with frame_buffer.lock:
            if frame_buffer.idx > idx_start and frame_buffer.frame is not None:
                return frame_buffer.frame.copy()
        time.sleep(0.006)
    with frame_buffer.lock:
        return frame_buffer.frame.copy() if frame_buffer.frame is not None else None

def run_round():
    with frame_buffer.lock:
        start_idx = frame_buffer.idx
        time.sleep(0.2)
    frame_bgr = capture_next_frame_after(start_idx)
    if frame_bgr is None:
        return  #nessun frame

    roi_rgb, box, ok = detect_and_crop(frame_bgr)
    if not ok:
        #mano non trovata → ignora il round
        return #mano non individuata
    
    #ritaglio rettangolo debug e salvo snapshot
    x1,y1,x2,y2 = box
    disp = frame_bgr.copy()
    cv2.rectangle(disp,(x1,y1),(x2,y2),(0,255,140),3)
    st.session_state.snapshot_bgr = disp
    st.session_state.snapshot_ready=True
    
    move, conf = predict_move(roi_rgb) #avvio predizione su immagine
    st.session_state.user_move = move
    st.session_state.conf = conf

    if move != "uncertain":
        cpu = random.choice(CLASS_NAMES); st.session_state.cpu_move = cpu
        res = decide(move, cpu); st.session_state.result = res
        if res == "win": st.session_state.score_u += 1
        elif res == "lose": st.session_state.score_c += 1
    else:
        st.session_state.cpu_move = ""; st.session_state.result = ""


st.markdown("<div class='layout-row'>", unsafe_allow_html=True)
left_col, right_col = st.columns([1/3, 2/3])

with left_col:
    st.markdown("""<div id='cam-wrapper'>
        <div id='snapshot-layer'></div>
        <div id='countdown-layer'></div>
    </div>""", unsafe_allow_html=True)

    webrtc_streamer(
        key="rps-fixed-small",
        video_frame_callback=video_frame_callback,
        rtc_configuration=RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]}),
        media_stream_constraints={"video": {"width": 300, "height": 200}},
        async_processing=True
    )

    bcol1, bcol2 = st.columns(2)
    start_round = bcol1.button("ROUND", disabled=st.session_state.round_running)
    reset_btn   = bcol2.button("RESET")

    if start_round and not st.session_state.round_running:
        st.session_state.round_running = True
        st.session_state.snapshot_ready = False
        st.session_state.user_move=""; st.session_state.cpu_move=""; st.session_state.result=""
        run_round()
        st.session_state.round_running = False
        if st.session_state.snapshot_ready and st.session_state.snapshot_bgr is not None:
            bgr = st.session_state.snapshot_bgr; rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            _, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            b64 = base64.b64encode(buf.tobytes()).decode()
            st.markdown(f"<script>const snap=document.getElementById('snapshot-layer'); if(snap){{snap.innerHTML='<img src=\"data:image/png;base64,{b64}\">';}}</script>", unsafe_allow_html=True)
        render_score()  # aggiorna punteggio immediatamente

    if reset_btn:
        st.session_state.score_u = 0; st.session_state.score_c = 0
        st.session_state.user_move = st.session_state.cpu_move = st.session_state.result = ""
        st.session_state.conf = 0.0; st.session_state.snapshot_ready = False
        render_score()

with right_col:
    st.markdown("<div class='result-area'>", unsafe_allow_html=True)
    user_move = st.session_state.user_move
    cpu_move = st.session_state.cpu_move
    conf = st.session_state.conf
    result = st.session_state.result

    move_display_map = {"rock":"✊ ROCK","paper":"🖐 PAPER","scissors":"✌️ SCISSORS","uncertain":"–––"}
    st.markdown("<div class='sep'></div>", unsafe_allow_html=True)
    st.markdown("<div class='big-moves'>", unsafe_allow_html=True)
    st.markdown(f"""<div class='card-move'> <div class='move-label'>YOU</div>
        <h2>{move_display_map.get(user_move,'–––')}</h2>
        <div class='confidence'>{f"{conf*100:.0f}%" if user_move and user_move!='uncertain' else ""}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class='card-move'>
        <div class='move-label'>CPU</div>
        <h2>{move_display_map.get(cpu_move,'–––')}</h2>
        <div class='confidence'></div>
    </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    banner_html = ""
    if result == "win":
        banner_html = "<div class='banner win'>YOU WIN!</div>"
    elif result == "lose":
        banner_html = "<div class='banner lose'>YOU LOSE!</div>"
    elif result == "draw":
        banner_html = "<div class='banner draw'>DRAW</div>"
    elif user_move == "uncertain" and user_move != "":
        banner_html = "<div class='banner draw' style='color:#bbb;'>UNCERTAIN</div>"
    
    st.markdown(banner_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)