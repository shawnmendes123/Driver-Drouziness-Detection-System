import cv2
import time
import csv
import numpy as np
from datetime import datetime
from pygame import mixer
import threading
import pygame
from queue import Queue, Empty

# -----------------------------
# Global exit flag
# -----------------------------
EXIT_REQUESTED = False

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
# PYGAME-based Animation3D Class
# -----------------------------
class Animation3D(threading.Thread):
    def __init__(self, width=640, height=480, fps=60):
        super().__init__(daemon=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.queue = Queue()
        self.running = False

        # visual state
        self.state = "normal"
        self.indicator_on = False
        self.shift = 0.0
        self.speed = 100.0
        self.max_shift = int(self.width * 0.24)

        # timers
        self.phase_start = None

        # durations (tweakable)
        self.indicator_duration = INDICATOR_DURATION
        self.lane_change_duration = LANE_CHANGE_DURATION
        self.decel_duration = DECEL_DURATION
        self.accel_duration = ACCEL_DURATION
        self.return_duration = RETURN_DURATION

    def send_command(self, cmd: dict):
        self.queue.put(cmd)

    def _process_commands(self):
        try:
            while True:
                cmd = self.queue.get_nowait()
                if not isinstance(cmd, dict): 
                    continue
                c = cmd.get("cmd", None)
                if c == "shutdown":
                    self.running = False
                elif c == "set_state":
                    s = cmd.get("state")
                    if s:
                        self._enter_state(s)
                elif c == "force":
                    if "shift" in cmd:
                        self.shift = float(cmd["shift"])
                    if "speed" in cmd:
                        self.speed = float(cmd["speed"])
        except Empty:
            pass

    def _enter_state(self, s):
        if s == self.state:
            return
        self.state = s
        self.phase_start = time.time()
        # optionally set indicator
        if s == "indicator":
            self.indicator_on = True
        elif s == "changing_right":
            self.indicator_on = True
        else:
            self.indicator_on = False

    def run(self):
        global EXIT_REQUESTED
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("3D Road Animation - Driver Assist")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 20, bold=True)

        # pre-draw car surface
        CAR_W, CAR_H = 140, 80
        car_surf = pygame.Surface((CAR_W, CAR_H), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, (10,140,200), (0, 12, CAR_W, CAR_H-12), border_radius=12)
        pygame.draw.rect(car_surf, (0,90,140), (14, -6, CAR_W-28, CAR_H-22), border_radius=10)
        pygame.draw.rect(car_surf, (255,240,200), (CAR_W-18, CAR_H//2 - 6, 8, 12), border_radius=3)

        last_time = time.time()
        self.running = True
        while self.running and not EXIT_REQUESTED:
            # Exit handling (allow closing window)
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                    EXIT_REQUESTED = True

                # ✅ Stop animation if X key is pressed
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_x:
                        self.running = False
                        EXIT_REQUESTED = True

            # handle external commands
            self._process_commands()

            now = time.time()
            dt = now - last_time
            last_time = now

            # State machine animation
            if self.state == "indicator":
                t = now - (self.phase_start or now)
                # indicator blinks
                self.indicator_on = (int(now * 2) % 2 == 0)
                # speed holds
                self.speed = 100.0
                # after indicator duration -> move to changing_right
                if t >= self.indicator_duration:
                    self._enter_state("changing_right")

            elif self.state == "changing_right":
                t = now - (self.phase_start or now)
                progress = clamp(t / self.lane_change_duration, 0.0, 1.0)
                smooth = ease_out_quad(progress)
                self.shift = self.max_shift * smooth
                # decelerate
                decel_prog = clamp(t / self.decel_duration, 0.0, 1.0)
                self.speed = 100.0 * (1.0 - decel_prog)
                if progress >= 1.0 and self.speed <= 1.0:
                    self.shift = float(self.max_shift)
                    self.speed = 0.0
                    self._enter_state("stopped")
                    # log via queue to main thread if wanted (not implemented here)
            elif self.state == "stopped":
                self.shift = float(self.max_shift)
                self.speed = 0.0
                self.indicator_on = False
                # wait until main thread commands 'returning' when driver wakes
            elif self.state == "returning":
                t = now - (self.phase_start or now)
                progress = clamp(t / self.return_duration, 0.0, 1.0)
                smooth = ease_out_quad(progress)
                start_shift = getattr(self, "_return_start_shift", self.max_shift)
                start_speed = getattr(self, "_return_start_speed", 0.0)
                # interpolate shift back to 0
                self.shift = start_shift * (1.0 - smooth)
                # accelerate
                accel_prog = clamp(t / self.accel_duration, 0.0, 1.0)
                self.speed = 100.0 * accel_prog
                if progress >= 1.0:
                    self.shift = 0.0
                    self.speed = 100.0
                    self._enter_state("normal")
                    self.running = self.running  # keep running, just go to normal
            else:  # normal
                # gently ensure values are nominal
                self.shift += (0.0 - self.shift) * min(1.0, dt * 4.0)
                self.speed += (100.0 - self.speed) * min(1.0, dt * 1.5)
                self.indicator_on = False

            # DRAWING
            screen.fill((20, 20, 30))
            # draw road perspective (simple trapezoid)
            van_y = int(self.height * 0.18)
            road_top_w = int(self.width * 0.28)
            road_bottom_w = int(self.width * 0.9)
            top_left = ((self.width - road_top_w) // 2, van_y)
            top_right = (top_left[0] + road_top_w, van_y)
            bot_left = ((self.width - road_bottom_w) // 2, self.height)
            bot_right = (bot_left[0] + road_bottom_w, self.height)
            pygame.draw.polygon(screen, (50,50,55), [top_left, top_right, bot_right, bot_left])

            # lane markers (draw converging lines)
            lane_count = 3
            for i in range(1, lane_count):
                lerp = i / lane_count
                sx = int(top_left[0] + (top_right[0]-top_left[0]) * lerp)
                ex = int(bot_left[0] + (bot_right[0]-bot_left[0]) * lerp)
                # dashed: draw segments
                seg_count = 18
                for s in range(seg_count):
                    a = s / seg_count
                    b = (s + 0.6) / seg_count
                    ay = int(van_y + (self.height - van_y) * a)
                    by = int(van_y + (self.height - van_y) * b)
                    ax = int(sx + (ex - sx) * a)
                    bx = int(sx + (ex - sx) * b)
                    pygame.draw.line(screen, (220,220,220), (ax, ay), (bx, by), 4)

            # right shoulder
            shoulder_x = int(bot_right[0] - (self.width * 0.02))
            pygame.draw.rect(screen, (80,80,80), (shoulder_x, 0, self.width - shoulder_x, self.height))

            # car position (base center of road bottom, then apply shift)
            car_base_x = self.width // 2
            car_x = int(car_base_x + self.shift - (car_surf.get_width() // 2))
            car_y = int(self.height * 0.65)
            screen.blit(car_surf, (car_x, car_y))

            # indicator arrow if on
            if self.indicator_on:
                # draw simple arrow near front-right of car
                arrow_color = (255, 190, 40) if (int(now*2) % 2 == 0) else (120,80,0)
                points = [(car_x + CAR_W - 10, car_y + 10),
                          (car_x + CAR_W + 40, car_y + 30),
                          (car_x + CAR_W - 10, car_y + 50)]
                pygame.draw.polygon(screen, arrow_color, points)

            # HUD: state text
            state_surf = font.render(self.state.upper(), True, (235,235,235))
            screen.blit(state_surf, (10, 10))

            # speed HUD
            speed_text = f"Speed: {int(self.speed):03d} km/h"
            speed_surf = font.render(speed_text, True, (255,255,255))
            sx = (self.width - speed_surf.get_width()) // 2
            screen.blit(speed_surf, (sx, 10))

            pygame.display.flip()
            clock.tick(self.fps)

        pygame.quit()

# Instantiate and start animation thread
anim = Animation3D(width=CAM_W, height=CAM_H, fps=30)
anim.start()

# -----------------------------
# Main loop
# -----------------------------
try:
    while True:
        # break if global exit requested by pygame thread or user
        if EXIT_REQUESTED:
            break

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
                    anim.send_command({"cmd": "set_state", "state": "indicator"})
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
                    # tell animation to return: capture current shift & speed in animation thread by forcing values
                    anim.send_command({"cmd": "set_state", "state": "returning"})
                    # also capture start shift/speed in animation thread (it reads current shift/speed internally)
                else:
                    # ensure normal state
                    animation_state = "normal"
                    animation_active = False
                    anim.send_command({"cmd": "set_state", "state": "normal"})

            COUNTER_START = None

        # Save video frames if recording
        if VIDEO_RECORDING and out is not None:
            out.write(frame)

        # Animation state transitions driven by main loop times
        # When animation was started, switch to changing_right after INDICATOR_DURATION
        if animation_active and animation_state == "indicator":
            if time.time() - animation_start_time >= INDICATOR_DURATION:
                animation_state = "changing_right"
                animation_start_time = time.time()
                anim.send_command({"cmd": "set_state", "state": "changing_right"})
        # when changing_right completes (we approximate by LANE_CHANGE_DURATION), tell anim to stop
        if animation_active and animation_state == "changing_right":
            if time.time() - animation_start_time >= LANE_CHANGE_DURATION:
                animation_state = "stopped"
                anim.send_command({"cmd": "set_state", "state": "stopped"})
                # log stopped
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([datetime.now().strftime("%H:%M:%S"), "AutoStoppedOnShoulder", ""])

        # Show windows
        cv2.imshow("Driver Drowsiness Detection", frame)
        # Note: animation runs in its own Pygame window

        key = cv2.waitKey(1) & 0xFF

        # ✅ Quit if user presses X in OpenCV window
        if key == ord("x"):
            EXIT_REQUESTED = True
            break

        # existing q key
        if key == ord("q"):
            EXIT_REQUESTED = True
            break

        # quit if pygame triggered shutdown
        if EXIT_REQUESTED:
            break

finally:
    # cleanup
    # tell animation thread to shutdown
    try:
        anim.send_command({"cmd": "shutdown"})
    except:
        pass
    # set global exit to ensure threads stop
    EXIT_REQUESTED = True
    anim.running = False
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    mixer.quit()
    # join anim thread briefly
    try:
        anim.join(timeout=1.0)
    except:
        pass
