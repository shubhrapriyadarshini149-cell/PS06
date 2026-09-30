# AI Factory Safety Monitoring System: 5-Day Implementation Blueprint

A production-grade, multi-camera computer vision edge pipeline designed for real-time factory compliance monitoring, including Personal Protective Equipment (PPE) compliance detection, fire/smoke hazard alerting, personnel proximity tracking, and multi-channel notification.

---

## 1. System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Ingestion ["1. Threaded Video Ingestion"]
        C1["Camera 1 (RTSP/MP4)"] -->|Reader Thread| Q1["Queue 1 (maxsize=1)"]
        C2["Camera 2 (RTSP/MP4)"] -->|Reader Thread| Q2["Queue 2 (maxsize=1)"]
        C3["Camera 3 (Webcam/MP4)"] -->|Reader Thread| Q3["Queue 3 (maxsize=1)"]
    end

    subgraph CoreEngine ["2. Core Processing Pipeline (Per Frame)"]
        Q1 & Q2 & Q3 --> Batcher["Frame Ingest & Frame Skip Controller"]
        Batcher --> Track["Ultralytics ByteTrack (Person Persistence)"]
        
        Track --> DetPPE["PPE Model (YOLOv11)\n[Hardhat, Vest, Boots, Gloves]"]
        Track --> DetFire["Fire/Smoke Model (YOLOv8 D-Fire)\n[Fire, Smoke]"]
        
        DetPPE --> PPERules["PPE Geometric Association\n(Head / Torso / Hand / Foot Regions)"]
        DetFire --> FireRules["Hazard & Proximity Engine\n(Feet-to-Fire Distance + Threat Levels)"]
        
        PPERules --> Smoother["Temporal Debouncer\n(Sliding Window: M of N frames)"]
        FireRules --> FirePersist["Fire Persistence Filter\n(Consecutive 5 frames + Size Check)"]
    end

    subgraph Action ["3. Alert & Output Layer"]
        Smoother & FirePersist --> AlertMgr["Alert Manager\n(Deduplication & 60s Cooldown)"]
        AlertMgr --> SQLite[("SQLite Event DB (logs.db)")]
        AlertMgr --> Snapshots["Disk Snapshots (outputs/snapshots/)"]
        AlertMgr --> Telegram["Telegram Bot / Webhook"]
        SQLite & Snapshots --> Dashboard["Streamlit Safety Operator Dashboard"]
    end
```

---

## 2. Project Directory Structure

```text
PS06/
├── config/
│   ├── cameras.yaml            # Camera source definitions, zones, and ignore boxes
│   └── rules.yaml              # Zone-specific PPE compliance requirements
├── models/
│   ├── ppe_yolov11.pt          # Pretrained PPE weights (person + gear classes)
│   └── fire_yolov8.pt          # Pretrained D-Fire/Smoke detection weights
├── src/
│   ├── __init__.py
│   ├── detectors.py            # YOLO wrapper & canonical class mapping
│   ├── tracking.py             # ByteTrack integration & track state management
│   ├── ppe_rules.py            # Anatomical bounding-box geometric association
│   ├── fire_rules.py           # Fire persistence, ignore masking & worker proximity
│   ├── temporal.py             # Sliding-window debouncing filter
│   ├── stream.py               # Threaded RTSP/video reader with frame-dropping queue
│   ├── alerts.py               # Cooldown cache, SQLite logging & Telegram dispatcher
│   └── main.py                 # Multi-camera orchestrator & inference loop
├── tests/
│   ├── test_images/            # Edge case factory stills (low light, occlusion)
│   └── test_videos/            # Sample videos (PPE violation, welding, fire)
├── outputs/
│   ├── snapshots/              # Annotated incident frames
│   └── logs.db                 # SQLite incident database
├── dashboard.py                # Streamlit monitoring and compliance review dashboard
├── requirements.txt            # Python dependencies
└── README.md                   # Setup guide and evaluation metrics
```

---

## 3. Hardware Budgeting & Execution Strategy

| Hardware Target | Model Sizing | Frame-Skip Policy | Expected Performance |
| :--- | :--- | :--- | :--- |
| **NVIDIA GPU (RTX 3060 / 4060+)** | `yolo11s` (PPE) + `yolov8s` (Fire) | Process every 2nd frame | 3–4 Streams @ 25–30 FPS |
| **Entry GPU (GTX 1650 / T4 / Laptop)** | `yolo11n` (PPE) + `yolov8n` (Fire) | Process every 3rd frame | 2–3 Streams @ 18–25 FPS |
| **CPU Only (Intel Core i7 / Ryzen 7)** | `yolo11n` + `yolov8n` (FP32/ONNX) | Process every 3rd or 4th frame | 1–2 Streams @ 10–15 FPS |

---

## 4. Detailed 5-Day Implementation Roadmap

### Day 1: Model Ingestion, Class Normalization & PPE Geometric Association

#### Morning: Environment Setup & Detector Engine
1. **Initialize Environment & Dependencies:**
   ```bash
   pip install ultralytics opencv-python pyyaml pydantic requests
   ```
2. **Model Weight Loading & Class Inspection:**
   - Load both model weights via Ultralytics.
   - Print `model.names` to reveal actual class strings.
   - Implement `CLASS_MAP` in `src/detectors.py` to decouple downstream rules from weight-specific naming variations:
     ```python
     CLASS_MAP = {
         # PPE Model Mappings
         "helmet": "hardhat", "hard hat": "hardhat", "hardhat": "hardhat", "white-helmet": "hardhat",
         "vest": "vest", "safety vest": "vest", "safety-vest": "vest", "reflective-jacket": "vest",
         "gloves": "gloves", "safety gloves": "gloves",
         "boots": "boots", "safety boots": "boots", "shoes": "boots",
         "person": "person", "worker": "person",
         # Fire / Smoke Model Mappings
         "fire": "fire", "flame": "fire",
         "smoke": "smoke"
     }
     ```
3. **Detector Wrapper (`src/detectors.py`):**
   - Encapsulate inference to return a clean data format:
     ```python
     # Dict output per detection
     {"cls": "hardhat", "conf": 0.88, "bbox": [x1, y1, x2, y2]}
     ```

#### Afternoon: Anatomical Sub-Region PPE Logic
1. **Geometric Association Math (`src/ppe_rules.py`):**
   - For each detected `person` bounding box $[x_1, y_1, x_2, y_2]$ with width $w$ and height $h$:
     - **Head (Hardhat):** $x \in [x_1 - 0.10w, x_2 + 0.10w]$, $y \in [y_1 - 0.10h, y_1 + 0.28h]$
     - **Torso (Vest):** $x \in [x_1 - 0.10w, x_2 + 0.10w]$, $y \in [y_1 + 0.22h, y_1 + 0.68h]$
     - **Hands (Gloves):** $x \in [x_1 - 0.20w, x_2 + 0.20w]$, $y \in [y_1 + 0.35h, y_1 + 0.75h]$
     - **Feet (Boots):** $x \in [x_1 - 0.10w, x_2 + 0.10w]$, $y \in [y_1 + 0.78h, y_2 + 0.10h]$
2. **Intersection / Centroid Test:**
   - Compute centroid of detected PPE item: $C = \left(\frac{x_1 + x_2}{2}, \frac{y_1 + y_2}{2}\right)$.
   - An item is assigned to a worker if $C$ falls within that item's defined anatomical sub-region.
3. **Zone Compliance Config (`config/rules.yaml`):**
   ```yaml
   zones:
     welding_bay:
       required_ppe: ["hardhat", "vest", "gloves"]
     loading_dock:
       required_ppe: ["hardhat", "vest", "boots"]
     general_walkway:
       required_ppe: ["vest"]
   ```

> **Day 1 Milestone Check:**
> Running `python detect_image.py tests/test_images/worker.jpg` renders an annotated image showing bounding boxes around each worker with an overlay label such as: `Status: VIOLATION | Missing: [hardhat, vest]`.

---

### Day 2: Video Stream Ingestion, ByteTrack & Temporal Smoothing

#### Morning: Video Pipeline & ByteTrack Integration
1. **OpenCV Video Loop:**
   - Ingest MP4 factory video clips with `cv2.VideoCapture`.
2. **Ultralytics ByteTrack Hook (`src/tracking.py`):**
   - Track `person` instances across consecutive frames using `model.track(frame, persist=True, tracker="bytetrack.yaml")`.
   - Extract unique `track_id` for every person.
3. **Frame-Skip Optimization:**
   - Run detector and tracker every $K^{\text{th}}$ frame (e.g., $K=2$). For non-inference frames, retain previous bounding boxes and status to maintain stable display at full playback speed.

#### Afternoon: Temporal Violation Smoothing (Debounce Engine)
1. **The Flickering Problem:**
   - Worker turning sideways or momentary camera noise can cause a hardhat to be missed for 1–2 frames, generating false violation spikes.
2. **Sliding-Window Debouncer (`src/temporal.py`):**
   - For every active `track_id`, maintain a rolling deque of size $N=30$ frames (~1 second at 30 FPS).
   - Record compliance state per frame: `1` for violation, `0` for compliant.
   - **Trigger Condition:** Flag sustained violation only when:
     $$\frac{\sum_{i=1}^{N} \text{state}_i}{N} \ge 0.60 \quad (\text{e.g., } \ge 18 \text{ out of the last } 30 \text{ frames})$$
   - **Clear Condition:** Mark compliant when violation drops below $20\%$ over the last 30 frames.

> **Day 2 Milestone Check:**
> Running `python test_video.py tests/test_videos/cctv_walkway.mp4` shows consistent person track IDs (`ID: 3`) and steady violation flags without single-frame flickering alerts.

---

### Day 3: Fire & Smoke Hazard Detection & Personnel Proximity

#### Morning: Fire/Smoke Inference & Multi-Frame Persistence
1. **D-Fire Model Pipeline (`src/fire_rules.py`):**
   - Run the dedicated fire/smoke YOLO model on incoming frames.
2. **False Positive Suppression:**
   - **Size Threshold:** Ignore detections with box area $< 400 \text{ px}^2$ (filters out tiny distant specks/lighting glints).
   - **Persistence Counter:** Require fire or smoke detection in at least $5$ consecutive frames before confirming a hazard event.
   - **Exclusion / Ignore Masks:** Read rectangular ignore zones from `config/cameras.yaml` to suppress known hot zones (furnaces, welding stations):
     ```yaml
     ignore_zones:
       - [x1, y1, x2, y2]  # Bounding coordinates to ignore fire detections
     ```

#### Afternoon: Spatial Proximity & Hazard Threat Classification
1. **Ground Plane Proximity Metric:**
   - Use the worker's bottom center coordinate (feet position): $P_{\text{worker}} = \left(\frac{x_1 + x_2}{2}, y_2\right)$.
   - Measure Euclidean distance to fire bounding box center: $P_{\text{fire}} = \left(\frac{x_{f1} + x_{f2}}{2}, \frac{y_{f1} + y_{f2}}{2}\right)$.
   - Normalize distance against fire box width $W_{\text{fire}} = x_{f2} - x_{f1}$:
     $$D_{\text{norm}} = \frac{\|P_{\text{worker}} - P_{\text{fire}}\|_2}{\max(W_{\text{fire}}, 50)}$$
2. **Threat Level Classification Matrix:**
   - **WARNING:** Smoke detected (regardless of worker proximity).
   - **HIGH:** Confirmed fire detected, but no workers within $D_{\text{norm}} \le 2.0$.
   - **CRITICAL:** Confirmed fire detected WITH $\ge 1$ worker within $D_{\text{norm}} \le 2.0$. Annotate the exact count of workers at risk.

> **Day 3 Milestone Check:**
> A fire test clip triggers a dynamic hazard banner: `HAZARD: CRITICAL | Fire Confirmed | 2 Workers at Risk | Distance: 1.2x`.

---

### Day 4: Multi-Stream Engine & Alert Dispatcher

#### Morning: Threaded Multi-Stream Ingestion (`src/stream.py`)
1. **Zero-Latency Ingestion Architecture:**
   - Standard OpenCV `cv2.VideoCapture` blocks and creates internal buffer lag on network streams.
   - Implement dedicated `StreamReader` thread per camera using `queue.Queue(maxsize=1)`.
   - **Frame Drop Logic:**
     ```python
     # Always keep only the newest frame
     if self.queue.full():
         try:
             self.queue.get_nowait()
         except queue.Empty:
             pass
     self.queue.put(frame)
     ```
2. **Resilience & Auto-Reconnect:**
   - If `cap.read()` fails (stream dropped), trigger exponential backoff reconnection loop (1s, 2s, 4s, up to 10s) without crashing the other camera pipelines.
3. **Multi-Camera Configuration (`config/cameras.yaml`):**
   ```yaml
   cameras:
     - id: "cam_weld_01"
       source: "tests/test_videos/welding.mp4"
       zone: "welding_bay"
     - id: "cam_dock_02"
       source: "tests/test_videos/dock.mp4"
       zone: "loading_dock"
     - id: "cam_walkway_03"
       source: "tests/test_videos/walkway.mp4"
       zone: "general_walkway"
   ```

#### Afternoon: Alert Dispatcher & Audit Storage (`src/alerts.py`)
1. **Cooldown & Deduplication Cache:**
   - Track key: `(camera_id, track_id, violation_type)`.
   - Suppress repeat notifications for $60$ seconds, **except** when a hazard escalates to `CRITICAL` (which bypasses cooldown immediately).
2. **SQLite Incident Logging (`outputs/logs.db`):**
   ```sql
   CREATE TABLE IF NOT EXISTS incident_logs (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       timestamp TEXT NOT NULL,
       camera_id TEXT NOT NULL,
       zone TEXT NOT NULL,
       incident_type TEXT NOT NULL, -- 'PPE_VIOLATION' | 'FIRE_HAZARD'
       track_id INTEGER,
       severity TEXT NOT NULL,      -- 'WARNING' | 'HIGH' | 'CRITICAL'
       details TEXT NOT NULL,
       snapshot_path TEXT NOT NULL
   );
   ```
3. **Telegram Notification Dispatch:**
   - When a sustained alert fires:
     - Save the annotated frame to `outputs/snapshots/<timestamp>_<cam_id>.jpg`.
     - Insert record into SQLite.
     - Dispatch Telegram bot photo message with formatted markdown payload asynchronously:
       ```text
       🚨 [CRITICAL SAFETY ALERT]
       Time: 2026-09-30 15:30:12
       Location: Welding Bay (Camera: cam_weld_01)
       Event: Fire Hazard - 2 Workers at Immediate Risk
       Snapshot attached.
       ```

> **Day 4 Milestone Check:**
> Running `python src/main.py` processes 3 video streams concurrently. A triggered PPE or fire violation automatically logs to SQLite, saves an annotated JPEG, and delivers an instant alert message on a mobile Telegram client.

---

### Day 5: Streamlit Dashboard, Benchmarking & Demo Delivery

#### Morning: Streamlit Operator Dashboard (`dashboard.py`)
1. **Lightweight Operational UI:**
   - **Top Metrics:** Total Violations Today, Critical Fire Events, Active Cameras, Average Compliance Rate ($94.2\%$).
   - **Live Incident Feed:** Auto-refreshing table from `outputs/logs.db` showing most recent 25 incidents with timestamp, zone, severity, and status.
   - **Snapshot Inspector:** Click on any incident row to inspect the stored full-resolution annotated snapshot image.
   - **Compliance Breakdown:** Bar chart showing compliance percentage broken down by factory zone.

#### Midday: Benchmarking & Performance Validation
1. **FPS & Latency Audit:**
   - Benchmark throughput with 1, 2, and 3 simultaneous streams.
   - Measure average end-to-end latency from frame ingest to alert dispatch (Target: $< 150 \text{ ms}$).
2. **Failure Mode Stress Test:**
   - Test low-light conditions, workers carrying bulky tools, and partial body occlusions.
   - Document any limitations in a failure analysis table for the final report.

#### Afternoon: Final Demo Packaging & Deliverables
1. **Record Showcase Demo Video:**
   - Scene 1: Worker entering zone without hardhat -> system tracks ID -> triggers alert after 1s -> snapshot taken.
   - Scene 2: Worker puts on hardhat -> status transitions to COMPLIANT -> alert resets.
   - Scene 3: Fire simulation clip -> triggers warning -> worker approaches -> escalates to CRITICAL -> Telegram notification arrives on phone.
2. **Documentation:**
   - Finalize `README.md` with installation commands, config reference, and quickstart scripts.

---

## 5. Configuration Files Reference

### `config/cameras.yaml`
```yaml
cameras:
  - id: "cam_dock_01"
    name: "Loading Dock North"
    source: "tests/test_videos/dock.mp4" # Or "rtsp://admin:pass@192.168.1.100:554/live"
    zone: "loading_dock"
    fps_limit: 25
    ignore_zones: []

  - id: "cam_weld_02"
    name: "Welding Cell 3"
    source: "tests/test_videos/welding.mp4"
    zone: "welding_bay"
    fps_limit: 25
    ignore_zones:
      - [340, 180, 480, 320] # Ignore fixed automated welding torch sparks
```

### `config/rules.yaml`
```yaml
zones:
  welding_bay:
    required_ppe: ["hardhat", "vest", "gloves"]
    min_confidence: 0.40
    smoothing_window: 30
    violation_threshold: 0.60

  loading_dock:
    required_ppe: ["hardhat", "vest", "boots"]
    min_confidence: 0.40
    smoothing_window: 30
    violation_threshold: 0.60

  general_walkway:
    required_ppe: ["vest"]
    min_confidence: 0.35
    smoothing_window: 25
    violation_threshold: 0.50
```

---

## 6. Risk Matrix & Production Mitigations

| Risk / Failure Mode | Likelihood | Impact | Built-in Mitigation |
| :--- | :--- | :--- | :--- |
| **Model Class Mismatch** | High | High | `CLASS_MAP` normalization dictionary created immediately on Day 1 morning. |
| **Flickering Alerts & Spam** | High | High | ByteTrack ID tracking combined with $M$-of-$N$ sliding window debouncing. |
| **Live Stream Buffer Delay** | High | Critical | Single-slot `Queue(maxsize=1)` frame dropping ensures zero lag accumulation. |
| **Welding Spark False Alarms** | Medium | Medium | Minimum box size filter ($>400\text{ px}^2$), 5-frame persistence, and rectangular ignore zones. |
| **Network Camera Disconnect** | High | Medium | Non-blocking reconnect loop with backoff in individual stream threads. |
| **Schedule Overrun** | Medium | Medium | Clear fallback tiers: drop dashboard if behind; focus on core pipeline and video demo. |

---

## 7. Pre-Flight Preparation (Day 0 Checklist)

Before commencing Day 1:
- [ ] Download PPE detection weights into `models/ppe_yolov11.pt`.
- [ ] Download Fire/Smoke detection weights into `models/fire_yolov8.pt`.
- [ ] Collect 3–4 test MP4 clips:
  - 1 PPE compliant worker video.
  - 1 PPE non-compliant worker video.
  - 1 Fire / smoke video.
  - 1 Challenging scenario (welding, sparks, or low lighting).
- [ ] Create a Telegram Bot via `@BotFather`, obtain `TELEGRAM_BOT_TOKEN`, and retrieve target `CHAT_ID`.
