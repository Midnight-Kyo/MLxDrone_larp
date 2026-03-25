# Hand Gesture Drone Project -- The Full Journey

> A documented walkthrough of building a real-time hand gesture recognition system from scratch, including every mistake, fix, and lesson learned along the way.

---

## The Goal

Build a system that recognizes hand gestures (two fingers, fist, open palm, thumbs up) from a live webcam feed in real-time, eventually to control a DJI Tello drone. No prior machine learning experience -- learning everything from the ground up.

---

## Phase 1: Environment Setup

### What we did

Set up a Python development environment on **WSL2 (Ubuntu 22.04)** with an **NVIDIA RTX 3070** GPU.

- Created a Python virtual environment (`venv`) to isolate project dependencies
- Detected the GPU and installed **PyTorch 2.10 with CUDA 12.8** for GPU-accelerated training
- Installed supporting libraries: OpenCV, Ultralytics (YOLOv8), djitellopy, NumPy, Matplotlib
- Verified every installation with test scripts
- Created the project folder structure:

```
gesture_drone/
├── dataset/
│   ├── train/      # 80% of collected images
│   └── val/        # 20% of collected images
├── dataset_cropped/ # (added later -- hand-only crops)
│   ├── train/
│   └── val/
├── models/          # Trained model weights
├── scripts/         # All Python scripts
└── README.md
```

### Trouble: `venv` module not available

The `python3 -m venv` command failed because `ensurepip` wasn't installed on the system.

**Fix:** Installed `python3.10-venv` via apt.

### What I learned

- **Virtual environments** keep project dependencies isolated so different projects don't conflict
- **CUDA** is NVIDIA's framework that lets PyTorch use the GPU's thousands of cores for parallel computation
- **PyTorch** is the deep learning framework; **torchvision** provides pre-trained models and image utilities
- **NumPy** handles array math; **Matplotlib** creates plots and charts
- The RTX 3070 with 8GB VRAM and 5,888 CUDA cores is solid for training small-to-medium models

---

## Phase 2: Data Collection

### What we did

Built `collect_data.py` -- a Python script that opens the webcam, shows a live feed, and captures hand gesture images at 10 frames per second when you press SPACE.

- 4 gesture classes: `two_fingers`, `fist`, `open_palm`, `thumbs_up`
- Press 1/2/3/4 to select the gesture, SPACE to start/stop recording, Q to quit
- Automatic 80/20 train/val split (random per frame)
- HUD overlay showing current gesture, recording status, and frame counts

### Trouble: WSL2 can't access the webcam

Running the script in WSL2 showed no video devices (`/dev/video`* didn't exist). WSL2 doesn't have native webcam access.

**Fix:** Ran the data collection script from **Windows PowerShell** instead, saving images directly to the WSL2 filesystem via the `\\wsl$\` network path. Training would still happen in WSL2 with GPU access.

### Trouble: Wrong camera selected (virtual cameras)

`cv2.VideoCapture(0)` kept opening "EOS Webcam Utility" (a Canon virtual camera) instead of the actual Insta360 Link 2 Pro webcam. The system had multiple virtual cameras (Canon EOS, NVIDIA Broadcast, Camo).

**Fix (attempt 1):** Added a camera scanner that tried indices 0-4 with different backends and showed live previews. Found 3 cameras producing frames, but none were obviously the Insta360.

**Fix (attempt 2):** Discovered the **Insta360 Link Controller** software was holding exclusive access to the camera. After fully quitting the Insta360 Link Controller app and trying camera index 2 with the MSMF backend, the real camera appeared.

**Final solution:** Simplified the script to accept a camera index as a command-line argument:

```powershell
python "\\wsl$\...\collect_data.py" 2
```

### The collection process

Collected images in structured "rounds" for each gesture, varying:

- Hand distance (close, medium, far)
- Hand angle (straight, tilted left/right, rotated)
- Hand position in frame (center, left, right, high, low)
- Both left and right hand
- Accessories (with/without glasses, with/without headset)

Camera was set to 720p at 30fps to match the Tello drone's camera specs.

### Final dataset


| Gesture     | Train     | Val     | Total     |
| ----------- | --------- | ------- | --------- |
| two_fingers | 986       | 239     | 1,225     |
| fist        | 510       | 108     | 618       |
| open_palm   | 565       | 150     | 715       |
| thumbs_up   | 663       | 153     | 816       |
| **TOTAL**   | **2,724** | **650** | **3,374** |


### What I learned

- Data collection is foundational -- garbage in, garbage out
- Variety in the dataset (angles, distances, lighting) prevents the model from memorizing specific scenes
- The 80/20 split ensures we have unseen images to test against
- `two_fingers` ended up with almost double the images of `fist` -- this imbalance would cause problems later

---

## Phase 3: First Training Attempt

### What we did

Built `train_model.py` using **transfer learning** with **EfficientNet-B0**:

- **EfficientNet-B0**: A CNN (Convolutional Neural Network) pre-trained on 1.2 million ImageNet images. It already knows how to detect visual features (edges, textures, shapes). We replaced its final classifier layer (1,000 ImageNet classes) with a new one (our 4 gestures).
- **Two-phase training:**
  - **Phase 1 (5 epochs):** Froze the entire backbone (4M+ parameters locked), only trained the new classifier head (5,124 parameters). This prevents the randomly-initialized head from corrupting the pre-trained features.
  - **Phase 2 (15 epochs):** Unfroze everything and fine-tuned the full model with a 10x smaller learning rate.
- **Data augmentation:** Random flips, rotations, brightness changes, and crops applied on-the-fly during training to increase variety without collecting new images.

### Training results

```
Phase1 [1/5]   train_acc=81.3%   val_acc=93.1%
Phase1 [5/5]   train_acc=98.8%   val_acc=96.3%
Phase2 [1/15]  train_acc=99.8%   val_acc=100.0%  ← hit 100% immediately
Phase2 [15/15] train_acc=100.0%  val_acc=100.0%
```

Training completed in **3.7 minutes** on the RTX 3070. Model saved to `models/gesture_model.pt` (16 MB).

**100% validation accuracy.** Sounds amazing, right?

### Trouble: The model failed completely in real-time

Built `test_model.py` to run the model on the live webcam. Results:


| What I showed       | Model predicted | Confidence |
| ------------------- | --------------- | ---------- |
| No hand (just face) | two_fingers     | 82.4%      |
| Fist                | two_fingers     | 43.5%      |
| Open palm           | two_fingers     | 81.3%      |
| Thumbs up           | thumbs_up       | 74.3%      |


The model predicted "two_fingers" for almost everything, including when no hand was visible at all.

### The diagnosis: three problems

**1. Class imbalance.** `two_fingers` had 1,225 images vs `fist` with 618. The model learned a shortcut: "when in doubt, say two_fingers" since that's statistically correct most often.

**2. No "none" class.** With no hand visible, the model *had* to pick one of 4 classes. It defaulted to its most confident one.

**3. Background memorization.** The model wasn't looking at hand shapes -- it was looking at the background, face, chair, and blinds. The validation images came from the same session with the same scene, so 100% accuracy was just the model recognizing the scene, not the gestures.

### What I learned

- **100% validation accuracy can be meaningless** if the validation set is too similar to the training set
- The model was classifying based on ~80% background pixels and ~20% hand pixels
- Class imbalance lets the model cheat by always guessing the majority class
- The real test is always live inference in conditions different from training

---

## Phase 4: First Fix Attempt (Weighted Training)

### What we did

Three changes to the training script:

1. **WeightedRandomSampler:** Forces each training batch to draw equally from all 4 classes, so `two_fingers` can't dominate.
2. **Weighted CrossEntropyLoss:** Loss weights inversely proportional to class frequency -- getting `fist` wrong costs 1.34x more than `two_fingers`.
3. **RandomErasing augmentation:** Randomly blacks out rectangular patches during training, forcing the model to not rely on any single image region (like the background).

### Results

Training showed more balanced learning. Phase 1 had `two_fingers` as the *least* confident class initially (83%) instead of the most -- the weighting was working.

Live testing improved slightly:

- Two fingers vs thumbs up could now be distinguished somewhat
- But **fist and open palm were never detected** -- always predicted as two_fingers or thumbs_up
- No hand visible still predicted a gesture (but at lower confidence)

### The real problem

The fundamental issue wasn't class imbalance -- it was that **the hand was too small in the frame**. When the 1280x720 image gets resized to 224x224 for the model, the hand occupies maybe 15-20% of the pixels. The model couldn't distinguish fine hand shape differences at that scale. It was classifying based on arm position and body posture, not hand shape.

---

## Phase 5: The MediaPipe Fix (Detection + Classification Pipeline)

### The insight

Instead of feeding the entire frame to one model and hoping it figures out both *where* the hand is and *what* gesture it's making, we split the problem into two stages:

1. **Stage 1 -- Hand Detection:** Use Google's **MediaPipe HandLandmarker** to find the hand in the frame and crop it out
2. **Stage 2 -- Gesture Classification:** Feed *only the hand crop* to EfficientNet for classification

This is how professional computer vision systems work -- specialized models composed together, each doing what it's best at.

### What is MediaPipe?

MediaPipe is Google's open-source framework for real-time perception. The HandLandmarker model detects 21 landmark points on a hand (wrist, knuckles, fingertips) and requires no training -- it works out of the box. It runs on CPU in real-time.

```
The 21 landmarks:
 0: Wrist
 1-4: Thumb (CMC → MCP → IP → TIP)
 5-8: Index finger (MCP → PIP → DIP → TIP)
 9-12: Middle finger
 13-16: Ring finger
 17-20: Pinky
```

### The key insight: training data must match inference

If the model is going to receive cropped hand images at inference time, it must be *trained* on cropped hand images too. So we:

1. **Batch-cropped the existing dataset:** Ran MediaPipe on all 3,374 original images, detected the hand in each one, cropped it with 25% padding, and saved to `dataset_cropped/`. 3,258 out of 3,374 images (96.6%) had hands successfully detected.
2. **Retrained EfficientNet on the cropped dataset:** The model now only sees hands -- no faces, no chairs, no blinds. It has no choice but to learn actual hand shapes.
3. **Updated the inference pipeline:**

```
BEFORE:  Full frame → Resize → EfficientNet → prediction
         (hand = ~15% of pixels, background = ~85%)

AFTER:   Full frame → MediaPipe finds hand → Crop → Resize → EfficientNet → prediction
         (hand = ~100% of pixels)
```

### Final results


| What I showed       | Model predicted | Confidence |
| ------------------- | --------------- | ---------- |
| No hand (just face) | "No hand"       | --         |
| Two fingers         | two_fingers     | **99.9%**  |
| Fist                | fist            | **98.7%**  |
| Open palm           | open_palm       | ✓ works    |
| Thumbs up           | thumbs_up       | ✓ works    |


The fist -- which was *completely invisible* to every previous model version -- now gets detected at 98.7% confidence. "No hand" correctly shows when no hand is visible instead of guessing a random gesture.

FPS dropped from ~28 to ~20 (running two models per frame) but 20 FPS is still smooth enough for real-time drone control.

### What I learned

- **Composing specialized models is not "cheating"** -- it's correct engineering. Every production ML system does this.
- **Your training data must match your inference pipeline.** If inference crops the hand, training data must be cropped the same way.
- Using the detector to auto-crop training images is elegant -- MediaPipe processed 3,374 images in 90 seconds, saving hours of manual annotation.
- The difference between 15% hand pixels and 100% hand pixels is the difference between a model that can't distinguish a fist from a palm and one that does it at 99% confidence.

---

## The Final System

### Architecture

```
Webcam (Insta360 Link 2 Pro, 720p 30fps)
    │
    ▼
MediaPipe HandLandmarker (detects hand, provides bounding box)
    │
    ├── No hand found → Display "No hand"
    │
    └── Hand found → Crop hand region (with 25% padding)
                        │
                        ▼
                  EfficientNet-B0 (classifies gesture)
                        │
                        ▼
                  Display: gesture name + confidence + bounding box
```

### Scripts


| Script              | Purpose                                                  | Runs on  |
| ------------------- | -------------------------------------------------------- | -------- |
| `collect_data.py`   | Capture hand gesture images from webcam                  | Windows  |
| `crop_hands.py`     | Batch-crop hands from dataset using MediaPipe            | WSL2     |
| `train_model.py`    | Train EfficientNet-B0 on cropped hand images             | WSL2+GPU |
| `test_model.py`     | Real-time gesture recognition (MediaPipe + EfficientNet) | Windows  |
| `view_landmarks.py` | Visualize MediaPipe's 21 hand landmarks                  | Windows  |


### Key numbers

- **Dataset:** 3,374 images → 3,258 cropped hand images across 4 classes
- **Model:** EfficientNet-B0, 4 million parameters, 16 MB on disk
- **Training time:** ~2.5 minutes on RTX 3070
- **Inference:** ~50ms per frame on CPU (MediaPipe + EfficientNet combined), ~20 FPS
- **Accuracy:** 99.8% on validation set; works reliably in real-time on live webcam

---

## Concepts I Learned Along the Way


| Concept                                | What it means                                                                                |
| -------------------------------------- | -------------------------------------------------------------------------------------------- |
| **Virtual environment**                | Isolated Python environment so project dependencies don't conflict                           |
| **CUDA**                               | NVIDIA's framework for running parallel computations on GPU                                  |
| **Transfer learning**                  | Reusing a pre-trained model's knowledge instead of training from scratch                     |
| **CNN (Convolutional Neural Network)** | Neural network that slides filters across images to detect patterns                          |
| **EfficientNet**                       | A family of CNNs designed by Google that balance depth, width, and resolution efficiently    |
| **Forward pass**                       | Feeding an image through the network layers to get a prediction                              |
| **Loss function (CrossEntropy)**       | Measures how wrong the prediction is compared to the true label                              |
| **Backpropagation**                    | Calculates how much each weight contributed to the error, then adjusts them                  |
| **Epoch**                              | One complete pass through the entire training dataset                                        |
| **Learning rate**                      | How big of a step to take when adjusting weights (too big = overshoot, too small = slow)     |
| **Learning rate scheduler**            | Automatically reduces the learning rate when progress stalls                                 |
| **Data augmentation**                  | Random transforms (flips, rotations, brightness) applied during training to increase variety |
| **Class imbalance**                    | When some classes have far more training images than others, biasing the model               |
| **Overfitting**                        | When the model memorizes training data instead of learning generalizable patterns            |
| **Two-phase training**                 | Freeze backbone first (protect pre-trained features), then unfreeze for fine-tuning          |
| **MediaPipe**                          | Google's framework for real-time perception (hands, face, pose)                              |
| **Detection vs classification**        | Detection finds *where* objects are; classification identifies *what* they are               |
| **Model composition**                  | Using multiple specialized models together, each handling one part of the problem            |


---

## What's Next

Connect the gesture recognition system to a **DJI Tello drone** using the `djitellopy` library, mapping each gesture to a flight command:

- **Two fingers** → Take off / move forward
- **Fist** → Stop / hover
- **Open palm** → Land
- **Thumbs up** → Flip / trick

The gesture recognition pipeline is ready. The next chapter is making it fly.