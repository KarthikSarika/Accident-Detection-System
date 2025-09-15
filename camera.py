import cv2
import numpy as np
import os
from detection import AccidentDetectionModel
from dotenv import load_dotenv
from twilio.rest import Client
import threading
import simpleaudio as sa
import time
import smtplib
from email.message import EmailMessage
from datetime import datetime
from collections import deque

# Load environment variables
load_dotenv()

# Twilio creds
TWILIO_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM = os.getenv("TWILIO_FROM_NUMBER")
TWILIO_TO = os.getenv("EMERGENCY_NUMBER")

# Email creds
EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

client = Client(TWILIO_SID, TWILIO_AUTH)
font = cv2.FONT_HERSHEY_SIMPLEX
model = AccidentDetectionModel("model.json", "model_weights.h5")

stop_buzzer = False


def make_call():
    call = client.calls.create(
        twiml="""
            <Response>
                <Say loop="0">
                    Alert! An accident has been detected. Please take immediate action.
                </Say>
            </Response>
        """,
        to=TWILIO_TO,
        from_=TWILIO_FROM
    )
    print(f"📞 Call initiated! Call SID: {call.sid}")


def send_email(video_path):
    """Send email with attached accident video proof"""
    msg = EmailMessage()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    msg["Subject"] = "🚨 Accident Detected - Video Proof Attached"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_TO
    msg.set_content(f"An accident was detected on {now}. Please find the attached 1-minute video proof.")

    with open(video_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(video_path)

    msg.add_attachment(file_data, maintype="video", subtype="mp4", filename=file_name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print(f"📧 Email sent successfully to {EMAIL_TO}")
    except Exception as e:
        print(f"⚠️ Email sending failed: {e}")


def play_buzzer(max_duration=300, frequency=1000):
    global stop_buzzer
    fs = 44100
    t = np.linspace(0, 1, fs, False)
    tone = np.sin(frequency * t * 2 * np.pi)
    audio = (tone * 32767).astype(np.int16)

    start_time = time.time()
    while not stop_buzzer and (time.time() - start_time < max_duration):
        play_obj = sa.play_buffer(audio, 1, 2, fs)
        play_obj.wait_done()


def startapplication(video_path="accident.mp4"):
    global stop_buzzer
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠ ERROR: Cannot open video file: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    buffer_seconds = 2
    after_seconds = 3
    frame_buffer = deque(maxlen=fps * buffer_seconds)

    call_triggered = False
    recording = False
    video_writer = None
    accident_clip_path = "accident_proof.mp4"
    after_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # keep last 30 sec frames
        frame_buffer.append(frame.copy())

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))
        pred, prob = model.predict_accident(roi[np.newaxis, ...])

        if pred == "Accident":
            pct = round(prob[0][0] * 100, 2)
            cv2.putText(frame, f"{pred} {pct}%", (20, 30), font, 1, (0, 0, 255), 2)

            if pct >= 98.0 and not call_triggered:
                make_call()
                stop_buzzer = False
                threading.Thread(target=play_buzzer, daemon=True).start()

                # start writing video: 30s before + 30s after
                video_writer = cv2.VideoWriter(accident_clip_path,
                                               cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps, (width, height))
                # dump buffered frames (30s before)
                for f in frame_buffer:
                    video_writer.write(f)

                recording = True
                after_counter = 0
                print("✅ Call made, buzzer activated, recording proof (±30s)...")
                call_triggered = True
                 # 🚀 Send email immediately
                threading.Thread(target=send_email, args=(accident_clip_path,), daemon=True).start()

        # record 30s after
        if recording:
            video_writer.write(frame)
            after_counter += 1
            if after_counter >= (fps * after_seconds):
                recording = False
                video_writer.release()
                print("🎥 Accident proof video saved.")
                send_email(accident_clip_path)

        # Overlay hint
        if call_triggered and not stop_buzzer:
            cv2.putText(frame, "Press 'S' to stop buzzer", (20, 65), font, 0.7, (255, 255, 0), 2)

        cv2.imshow("Video", frame)

        key = cv2.waitKey(33) & 0xFF
        if key == ord("q"):
            stop_buzzer = True
            break
        elif key == ord("s") and call_triggered:
            stop_buzzer = True
            print("🔕 Buzzer stopped manually.")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    startapplication()
