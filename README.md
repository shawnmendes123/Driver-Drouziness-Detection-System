# Driver Drowsiness Detection + Autonomous Lane-Change Simulation

### **By Shawn Mendes**

**Mukesh Patel School of Technology Management and Engineering, NMIMS**

---

## 🚗 Project Overview

This project is an advanced **Driver Drowsiness Detection System** integrated with a **realistic autonomous lane‑change and safe‑stop simulation**, inspired by real-world ADAS (Advanced Driver Assistance Systems) used by Volkswagen, Tesla, Mercedes-Benz, and BMW.

The system:

* Detects driver eye closure using **OpenCV Haar cascades**
* Triggers **warnings, alarms, video recording, logging**
* Simulates an **autonomous right-lane pull-over** with indicator animation
* Gradually slows the vehicle from **100 km/h → 0 km/h**
* **Resumes driving** when the driver becomes alert again

This project is perfect for showcasing skills in:
✅ Computer Vision
✅ Python
✅ Real‑time systems
✅ Safety automation
✅ Human‑machine interaction
✅ Automotive engineering concepts

---

## 🧠 Features

### ✅ **1. Real-Time Drowsiness Detection**

* Face + eye detection using Haar cascades
* Detection of eye closure duration
* Countdown warnings
* Alarm activation after a threshold

### ✅ **2. Autonomous Lane-Change Simulation**

A separate OpenCV window simulates:

* Car driving in center lane
* Right indicator blinking
* Smooth lane shift animation
* Speed reduction from 100 → 0
* Full stop on the roadside
* Reverse animation when driver wakes up

### ✅ **3. Event Logging & Recording**

* CSV file logs drowsiness events
* Video of drowsiness incident is saved automatically

### ✅ **4. Emergency Recovery System**

When driver reopens eyes:

* Alarm stops
* Car accelerates back to 100 km/h
* Car returns to main lane

---

## 🧪 Technologies Used

* **Python**
* **OpenCV** (Computer Vision)
* **Pygame** (Alarm system)
* **Haar Cascades** (Face & eye detection)
* **CSV Logging**
* **avi Video Recording**

---

## 📁 Project Structure

```
│
├── main.py                     # Core drowsiness detection system
├── lane_simulation.py          # Animated lane-change simulation
├── drowsiness_log.csv          # Auto-generated event logs
├── sleep_*.avi                 # Auto-recorded video samples
├── assets/                     # Alarms, icons, indicators
└── README.md                   # Project documentation
```

---

## ▶️ How to Run

### **1. Install Dependencies**

```
pip install opencv-python pygame
```

### **2. Run the program**

```
python main.py
```

Make sure your webcam is connected.

---

## 🎯 Real-World Applications

This project replicates real automotive technologies like:

* Volkswagen Emergency Assist
* Tesla Autopilot Driver Monitoring
* Mercedes Attention Assist

Practical use cases:

* Driver Monitoring Systems (DMS)
* Accident prevention research
* ADAS prototyping
* Human‑vehicle safety studies
* Final year project & portfolio showcase

---

## 📌 Future Improvements

* CNN‑based eye state recognition
* Yawn detection
* Head pose estimation
* Infrared driver monitoring system (IR‑DMS)
* Vehicle CAN bus integration simulation

---

## 🏆 Author & Credits

**Shawn Mendes**
Mukesh Patel School of Technology Management and Engineering, NMIMS

This project was developed as a demonstration of advanced driver safety automation, combining computer vision and real-time vehicle behavior simulation.

---

## ⭐ If you like this project

Consider starring ⭐ the repository on GitHub!

# Drowsiness Detection System 🚗💤

## 📌 Overview

This project is a real-time **Driver Drowsiness Detection System** developed using **Python, OpenCV, MediaPipe, and EAR (Eye Aspect Ratio)** logic. It simulates a car dashboard where:

* The **speed decreases from 100 km/h to 0 km/h** when the driver becomes drowsy.
* A **warning alarm** triggers when the driver’s eyes remain closed for a defined threshold.
* When the system detects the driver looking back at the camera, the **car accelerates back and returns smoothly to the center**, simulating recovery.

This project demonstrates practical application of **computer vision**, **human‑computer interaction**, and **driver safety engineering**.

---

## 🧠 Features

✅ Real-time face & eye detection using **MediaPipe Face Mesh**
✅ EAR-based eye closure detection
✅ Dynamic speed display (100 → 0 depending on alertness)
✅ Warning alert sound on drowsiness
✅ Car movement simulation (normal → drift → recover)
✅ Smooth repositioning when driver looks back at the camera
✅ Highly customizable thresholds for sensitivity

---

## 🏎️ System Flow

1. Detect face → get eye landmarks
2. Compute EAR (Eye Aspect Ratio)
3. If EAR < threshold → eyes closed
4. After defined frames:

   * Decrease car speed
   * Car drifts sideways
   * Trigger alarm + danger state
5. Once EAR returns to normal → driver awake

   * Car accelerates back to 100
   * Car returns smoothly to center

---

## 🛠️ Technologies Used

* **Python 3**
* **OpenCV** – computer vision engine
* **MediaPipe** – facial landmark tracking
* **NumPy** – maths operations
* **Pygame** – warning sound

---

## 📂 Project Structure

```
📦 drowsiness-detection-system
 ┣ 📜 main.py
 ┣ 📁 assets/
 ┃ ┣ car.png
 ┃ ┗ alarm.wav
 ┣ 📜 README.md
 ┗ 📜 research-paper.pdf / docx
```

---

## ▶️ How to Run

### **1. Install dependencies:**

```
pip install opencv-python mediapipe pygame numpy
```

### **2. Run the program:**

```
python main.py
```

---

## 🧪 Real-World Applications

* Driver safety monitoring in cars/trucks
* Fleet management systems
* Automotive AI research
* Human–computer interaction projects
* Smart transportation engineering

---

## 👨‍💻 About the Developer

**Name:** Shawn Mendes
**University:** Mukesh Patel School of Technology Management and Engineering (NMIMS)

This project was developed as a practical exploration of computer vision systems, safety automation, and intelligent transportation modeling. The system integrates multiple AI models and algorithms—MediaPipe for real-time face tracking, EAR-based drowsiness measurement, and OpenCV visual simulation—to achieve a realistic driver‑monitoring experience.

---

## 📄 Research Paper

The full IEEE-format research paper is included in this repository as:

* `DrowsinessDetection_ShawnMendes.pdf`
* `DrowsinessDetection_ShawnMendes.docx`

For queries or collaborations:
**Shawn Mendes – NMIMS MPSTME**
