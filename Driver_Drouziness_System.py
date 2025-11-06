import cv2
import time
import csv
import numpy as np
from datetime import datetime
from pygame import mixer

# -----------------------------
# Initialize Alarm
# -----------------------------
mixer.init()
alarm_sound = "C:\\Users\\SHAWN\\Downloads\\Alarm Sound Effect.wav"  # Use .wav file (update path if needed)
mixer.music.load(alarm_sound)

# -----------------------------
# Constants / Tunables
# -----------------------------
EYE_CLOSED_TIME = 3       # seconds before alert
WARNING_TIME = 5          # countdown display on camera feed
VIDEO_FPS = 20

# Animation timing (tweakable)
INDICATOR_DURATION = 2.0      # seconds indicator blinks before lane change begins
LANE_CHANGE_DURATION = 4.0    # seconds to complete lane change (visual)
DECEL_DURATION = 3.0          # seconds to decelerate 100 -> 0 (medium)
ACCEL_DURATION = 3.0          # seconds to accelerate 0 -> 100 (medium)
RETURN_DURATION = 4.0         # seconds to complete return-to-center visual (can match LANE_CHANGE_DURATION)

# Visual sizes (pixels)
CAM_W = 640
CAM_H = 480

# -----------------------------
# Load Haar cascades
# -----------------------------
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# -----------------------------
# CSV logging
# -----------------------------
csv_file = "drowsiness_log.csv"
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Timestamp", "Event", "Duration(s)"])

# -----------------------------
# Camera init (try multiple backends)
# -----------------------------
cap = None
for api in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, None]:
    if api is not None:
        cap = cv2.VideoCapture(0, api)
    else:
        cap = cv2.VideoCapture(0)
    time.sleep(1.0)
    if cap.isOpened():
        print(f"Camera opened with backend: {api}")
        break
    else:
        cap.release()
        cap = None
if cap is None or not cap.isOpened():
    print("ERROR: Could not access the camera. Please check your webcam and try again.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)

# -----------------------------
# Video recording placeholders
# -----------------------------
out = None
video_start_time = None
VIDEO_RECORDING = False

# -----------------------------
# State variables
# -----------------------------
COUNTER_START = None
ALARM_ON = False

# Animation states: normal, indicator, changing_right, stopped, returning
animation_state = "normal"
animation_active = False
animation_start_time = None

# Movement and speed
max_shift_pixels = int(CAM_W * 0.24)  # how far right the car moves (about 24% of width)
current_shift = 0.0                   # current horizontal shift (pixels)
current_speed = 100.0                 # km/h, 100 normal -> 0 stopped
target_shift = 0.0

# Helper easing
def ease_out_quad(t):
    return 1 - (1 - t) * (1 - t)

def clamp(v, a, b):
    return max(a, min(b, v))

# -----------------------------
# Animation drawing function
# -----------------------------
def create_animation_frame(width, height, lane_shift, indicator_on, state_text, speed_value):
    anim = np.zeros((height, width, 3), dtype=np.uint8)

    # Road background
    cv2.rectangle(anim, (0, 0), (width, height), (40, 40, 40), -1)

    # Lane markers (vertical dashed)
    lane_left = int(width * 0.33)
    lane_right = int(width * 0.66)
    for y in range(0, height, 40):
        cv2.line(anim, (lane_left, y), (lane_left, y + 20), (200, 200, 200), 3)
        cv2.line(anim, (lane_right, y), (lane_right, y + 20), (200, 200, 200), 3)

    # Right shoulder
    shoulder_x = int(width * 0.9)
    cv2.rectangle(anim, (shoulder_x, 0), (width, height), (60, 60, 60), -1)
    cv2.putText(anim, "RIGHT SHOULDER", (shoulder_x - 170, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # Car drawing
    car_w, car_h = 110, 60
    center_x = width // 2
    base_x = int(center_x - car_w // 2)
    car_x = int(base_x + lane_shift)
    car_y = int(height * 0.65)

    # Body and roof
    cv2.rectangle(anim, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 120, 255), -1)
    cv2.rectangle(anim, (car_x + 15, car_y - 20), (car_x + car_w - 15, car_y + 10), (0, 90, 200), -1)
    # Wheels
    cv2.circle(anim, (car_x + 20, car_y + car_h), 10, (20, 20, 20), -1)
    cv2.circle(anim, (car_x + car_w - 20, car_y + car_h), 10, (20, 20, 20), -1)

    # Right indicator (blinking)
    ind_x1 = car_x + car_w - 6
    ind_y1 = car_y + 10
    ind_x2 = car_x + car_w + 6
    ind_y2 = car_y + 25
    ind_x1 = max(0, min(width - 1, ind_x1))
    ind_x2 = max(0, min(width - 1, ind_x2))
    ind_color = (0, 200, 255) if indicator_on else (30, 30, 30)
    cv2.rectangle(anim, (ind_x1, ind_y1), (ind_x2, ind_y2), ind_color, -1)

    # HUD: state_text top-left
    cv2.putText(anim, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)

    # SPEED display top-center
    speed_text = f"Speed: {int(speed_value):d} km/h"
    (tw, th), _ = cv2.getTextSize(speed_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    sx = (width - tw) // 2
    cv2.putText(anim, speed_text, (sx, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return anim

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("WARNING: Failed to grab frame or frame is empty.")
            time.sleep(0.5)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        eyes_detected = 0

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            eyes_detected = len(eyes)
            # draw for debug
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Drowsiness timer
        if eyes_detected == 0:
            if COUNTER_START is None:
                COUNTER_START = time.time()
            elapsed = time.time() - COUNTER_START
            remaining = max(0, WARNING_TIME - int(elapsed))
            cv2.putText(frame, f"Are you drowsy? {remaining}s", (400, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if elapsed >= EYE_CLOSED_TIME:
                # show alert box and text
                if int(time.time() * 2) % 2 == 0:
                    cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(frame, "DRIVER HAS SLEPT", (150, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                # start alarm if not already
                if not ALARM_ON:
                    mixer.music.play(-1)
                    ALARM_ON = True
                    # start animation sequence
                    animation_active = True
                    animation_start_time = time.time()
                    animation_state = "indicator"
                    # log
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([datetime.now().strftime("%H:%M:%S"), "AlarmStarted", ""])
                # start recording
                if not VIDEO_RECORDING:
                    video_start_time = time.time()
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    filename = datetime.now().strftime("sleep_%Y%m%d_%H%M%S.avi")
                    out = cv2.VideoWriter(filename, fourcc, VIDEO_FPS,
                                          (frame.shape[1], frame.shape[0]))
                    VIDEO_RECORDING = True
        else:
            # Eyes open -> reset countdown
            if VIDEO_RECORDING:
                duration = time.time() - video_start_time
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([datetime.now().strftime("%H:%M:%S"), "DrowsyEventDuration", f"{duration:.2f}"])
                VIDEO_RECORDING = False
                out.release()
                out = None

            if ALARM_ON:
                mixer.music.stop()
                ALARM_ON = False
                # When driver recovers, initiate return-to-center if we were not already normal
                if animation_active and animation_state in ("indicator", "changing_right", "stopped"):
                    # start returning sequence from current_shift & current_speed
                    animation_state = "returning"
                    animation_start_time = time.time()
                else:
                    # ensure normal state
                    animation_state = "normal"
                    animation_active = False

            COUNTER_START = None

        # Save video frames if recording
        if VIDEO_RECORDING and out is not None:
            out.write(frame)

        # -----------------------------
        # Animation state machine update
        # -----------------------------
        now = time.time()

        # Default flags
        indicator_on = False

        if animation_active:
            # If currently in indicator phase
            if animation_state == "indicator":
                t = now - animation_start_time
                # Blink indicator
                indicator_on = (int(now * 2) % 2 == 0)
                # Speed remains normal until lane change begins (optionally can start decel here)
                # If driver wakes up during indicator, we'll handle in above eyes_detected block which sets state to 'returning'
                if t >= INDICATOR_DURATION:
                    # proceed to lane change
                    animation_state = "changing_right"
                    # mark starting time for change & deceleration
                    animation_start_time = now
                    # store initial speed (should be 100)
                    start_speed_for_decel = current_speed

            elif animation_state == "changing_right":
                t = now - animation_start_time
                progress = clamp(t / LANE_CHANGE_DURATION, 0.0, 1.0)
                smooth = ease_out_quad(progress)
                current_shift = int(max_shift_pixels * smooth)

                # decelerate speed over DECEL_DURATION (start at changing start)
                decel_progress = clamp(t / DECEL_DURATION, 0.0, 1.0)
                current_speed = 100.0 * (1.0 - decel_progress)  # linear decel, you can ease if desired
                current_speed = clamp(current_speed, 0.0, 100.0)

                indicator_on = (int(now * 3) % 2 == 0)  # keep blinking faster while changing

                # if completed movement and speed nearly zero -> stopped
                if progress >= 1.0 and current_speed <= 1.0:
                    current_shift = max_shift_pixels
                    current_speed = 0.0
                    animation_state = "stopped"
                    # hold stopped until driver recovers
                    # log event
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([datetime.now().strftime("%H:%M:%S"), "AutoStoppedOnShoulder", ""])
            elif animation_state == "stopped":
                # hold stopped; speed stays 0 until driver wakes -> handled in eyes_detected change which switches to 'returning'
                current_shift = max_shift_pixels
                current_speed = 0.0
                indicator_on = False

            elif animation_state == "returning":
                # returning to center from current_shift
                t = now - animation_start_time
                progress = clamp(t / RETURN_DURATION, 0.0, 1.0)
                smooth = ease_out_quad(progress)
                # compute shift interpolating from current value at start to 0
                # But we need initial_shift at the moment return started - capture it by computing based on how far we are
                # To keep simple, compute shift_target by linear interpolation from wherever we currently are at return start.
                # We'll approximate by using current_shift as the start and easing toward 0:
                start_shift = current_shift if current_shift != 0 else max_shift_pixels  # fallback
                current_shift = int(start_shift * (1.0 - smooth))

                # accelerate speed up from current_speed (likely 0) to 100 over ACCEL_DURATION
                accel_progress = clamp(t / ACCEL_DURATION, 0.0, 1.0)
                current_speed = 100.0 * accel_progress
                current_speed = clamp(current_speed, 0.0, 100.0)

                indicator_on = False

                # When fully returned
                if progress >= 1.0 and current_shift <= 1:
                    current_shift = 0
                    current_speed = 100.0
                    animation_state = "normal"
                    animation_active = False
                    # log resume
                    with open(csv_file, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([datetime.now().strftime("%H:%M:%S"), "AutoReturnCompleted", ""])
        else:
            # Normal driving visuals
            current_shift = 0.0
            current_speed = 100.0
            animation_state = "normal"
            indicator_on = False

        # Make sure variables are numeric and clamped
        current_shift = float(clamp(current_shift, 0.0, max_shift_pixels))
        current_speed = float(clamp(current_speed, 0.0, 100.0))

        # Create animation frame
        anim_frame = create_animation_frame(CAM_W, CAM_H, lane_shift=int(current_shift),
                                            indicator_on=indicator_on, state_text=animation_state.upper(),
                                            speed_value=current_speed)

        # Show windows
        cv2.imshow("Driver Drowsiness Detection", frame)
        cv2.imshow("Car Animation", anim_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

finally:
    # cleanup
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    mixer.quit()
