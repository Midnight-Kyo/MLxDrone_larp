# Simulation Phase -- Where We Are Now

> A continuation of `SIMULATION_GUIDE.md`. This document captures the current state of the ROS2+Gazebo simulation, what works, what doesn't, and the elephant in the room: the gesture model.

---

## What We Can Do Right Now

The full end-to-end pipeline is operational:

1. **Windows side** -- `gesture_bridge.py` opens the webcam, runs MediaPipe hand detection + EfficientNet-B0 classification, and streams JSON commands over TCP to WSL2.
2. **WSL2 side** -- `gesture_ros2_node.py` receives those commands, translates them into ROS2 messages, and controls a simulated Tello drone in Gazebo.
3. **Gazebo** -- The drone lives in a "pretty world" with a sky and clouds, textured grass, landmark objects, and a camera that automatically tracks the drone as it flies.

### Working Commands

| Gesture | Command | Drone Behavior |
|---------|---------|----------------|
| Two fingers (peace sign) | `MOVE_FORWARD` | Takes off (if landed), then flies forward |
| Fist | `STOP` | Drone hovers in place |
| Open palm | `LAND` | Drone descends and lands |
| Thumbs up | `MOVE_UP` | Takes off (if landed), then ascends |

### How to Run It

**Terminal 1 (WSL2):**
```bash
cd /home/kyo/Projects/MLxDrone_larp/gesture_drone/scripts
bash launch_gazebo_bridge.sh
```

**Terminal 2 (Windows PowerShell):**
```powershell
python "\\wsl$\Ubuntu-22.04\home\kyo\Projects\MLxDrone_larp\gesture_drone\scripts\gesture_bridge.py" 2
```

---

## What Actually Works Well

- **The architecture is solid.** TCP bridge between Windows and WSL2 is reliable. ROS2 communication, topic publishing, service calls -- all working.
- **Gazebo rendering is GPU-accelerated.** After sorting out WSLg, the `GALLIUM_DRIVER=d3d12` + `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA` trick gives us RTX 3070 rendering in WSL2.
- **Takeoff and landing work.** The state machine in `gesture_ros2_node.py` correctly mirrors the Tello plugin's 4 states (landed, taking_off, flying, landing).
- **Stop command is reliable.** Fist → STOP works consistently. The drone holds position.
- **The pretty world renders.** Sky, clouds, grass, marker poles, landing pad, camera tracking -- all functional.

---

## What Doesn't Work Well

### 1. The Model (The Big Problem)

This is the bottleneck. Everything downstream -- the bridge, ROS2, Gazebo, the state machine -- is only as good as the gesture the model predicts. Right now, the model has three serious issues:

#### Problem A: Open Palm Ghost Predictions

The model frequently classifies non-open-palm gestures as `open_palm`. For example, showing the back of a two-finger pose gets misclassified as open palm. This triggers unwanted LAND commands during flight.

**Why this is happening:**

- The training data was collected by one person (you), in one environment, mostly from one angle.
- `open_palm` has 555 cropped training images. That sounds like enough, but those images likely share the same background, lighting, skin tone, and hand orientation.
- The model may have learned "fingers spread apart in front of that specific background" rather than "an open palm regardless of context."
- The back of a hand with two extended fingers looks somewhat like an open palm in silhouette -- similar spread of visible skin area.

#### Problem B: Rapid State Switching

The model flickers between predictions frame-to-frame. One frame says `two_fingers` at 88%, the next says `open_palm` at 82%, the next goes back to `two_fingers`. This causes the drone to receive rapid contradictory commands.

**What won't fix this alone:**

- **Temporal filtering** (e.g., requiring N consecutive frames of the same gesture): This helps if the model is "mostly right but occasionally wrong." It does NOT help if the model genuinely oscillates on borderline inputs. If the hand pose is ambiguous to the model, a temporal filter just adds latency without solving the root cause.
- **Raising the confidence threshold**: Currently at 85%. We could raise it to 95%, but then legitimate gestures at 80-88% confidence get ignored. The problem isn't that the model is uncertain -- it's that it's *confidently wrong*.

**What would actually fix this:**

More training data. Specifically, more *diverse* training data.

#### Problem C: Doesn't Generalize to Other People

The model works on your hands but struggles with your parents' hands. MediaPipe still detects the hand correctly (bounding box is good), but the EfficientNet classifier fails on the cropped hand image.

**Why:**

The model has only ever seen one person's hands. It has learned features that are entangled with your specific:
- Skin tone and texture
- Hand size and proportions
- Ring finger/pinky positioning habits
- Background and lighting conditions

This is a textbook **overfitting-to-the-data-collection-environment** problem.

---

## The Data Situation

### Current Dataset (what the model was trained on)

| Class | Cropped Train | Cropped Val | Total |
|-------|------------:|------------:|------:|
| fist | 510 | 108 | 618 |
| open_palm | 555 | 148 | 703 |
| thumbs_up | 587 | 140 | 727 |
| two_fingers | 973 | 237 | 1,210 |
| **Total** | **2,625** | **633** | **3,258** |

### What's Wrong With This Data

1. **Single person.** One skin tone, one hand shape, one set of habits. The model has never seen another human's hand.

2. **Single environment.** All images were captured from the same webcam, same desk, same lighting. The model may have learned background features as part of the gesture.

3. **Class imbalance.** `two_fingers` has almost 2x the images of `fist`. WeightedRandomSampler and class-weighted loss help, but they can't manufacture variety that doesn't exist.

4. **Missing viewpoints.** The training data likely lacks:
   - Back-of-hand views (which is exactly where open_palm ghost predictions happen)
   - Rotated hand angles
   - Different distances from camera
   - Different lighting conditions (bright, dim, side-lit)

5. **No hard negatives.** The model hasn't been shown "this looks like open_palm but it's actually two_fingers from behind" examples. Without these, it can't learn the subtle differences.

---

## What Would Fix the Model

### Path 1: Collect More Data (most impactful)

The single most effective thing is to collect more diverse training images:

- **Multiple people** -- different skin tones, hand sizes, ages
- **Multiple environments** -- different rooms, lighting, backgrounds
- **Multiple angles** -- front, back, side, rotated, tilted
- **Hard negatives** -- specifically collect the failure cases (back-of-hand two fingers, awkward open palms)

A realistic target: **500+ images per class from at least 3 different people in 2+ environments.**

The data collection script (`collect_data.py`) already exists and works. The cropping pipeline (`crop_hands.py`) also works. The workflow is:
1. Run `collect_data.py` on Windows with the webcam
2. Run `crop_hands.py` on WSL2 to create hand-only crops
3. Run `train_model.py` on WSL2 with GPU to retrain

### Path 2: Public Hand Gesture Datasets

Finding public datasets with exactly our 4 gestures (two fingers, fist, open palm, thumbs up) is tricky because most public datasets use different gesture sets. But options exist:

- **HaGRID** (HAnd Gesture Recognition Image Dataset) -- 552k images, 18 gesture classes. Contains `palm`, `fist`, `peace` (our two_fingers), `like` (our thumbs_up). Would need to filter, crop, and map to our classes.
- **ASL Alphabet datasets** -- some overlap but not a direct match.
- Custom scraping from video sources.

The challenge: public datasets may use different cropping, resolution, and hand detection approaches. Mixing them with our MediaPipe-cropped data requires careful preprocessing to avoid domain shift.

### Path 3: Data Augmentation (helps, but doesn't solve the core problem)

The training script already uses augmentation (random crop, flip, rotation, color jitter, random erasing). We could add:

- **Horizontal flip** -- already present, but note: flipping a thumbs up gives a thumbs up with the wrong hand, which is fine
- **Background replacement** -- after cropping, paste the hand onto random backgrounds
- **Skin tone augmentation** -- color shifts specifically targeting skin hue ranges
- **Cutout/CutMix** -- force the model to rely on hand shape rather than specific regions

But augmentation generates synthetic variety. It's not a substitute for real images of real different hands.

### Path 4: Temporal Filtering + Hysteresis (Software-side smoothing)

Even after improving the model, some amount of prediction noise is inevitable. A well-designed temporal filter would help:

- **Majority vote over a sliding window** (e.g., last 5 frames must agree)
- **Hysteresis** -- once a gesture is "locked in," require higher confidence to switch away from it
- **Dead zone** -- after a gesture change, ignore new gestures for N milliseconds

This is worth implementing regardless, but it should come AFTER the model is improved. Otherwise you're putting a bandaid on a broken leg.

---

## Remaining Simulation Issues (Not Model-Related)

### rc=3 Spam

The Tello plugin returns `rc=3` (BUSY) when commands arrive during state transitions (taking_off, landing). The state machine in `gesture_ros2_node.py` handles this, but rapid gesture switching from the model causes repeated service calls that get rejected. The fix is upstream (better model → fewer spurious commands).

### Gazebo Startup Time

Gazebo takes a while to load the pretty world on first launch. This is normal -- it's downloading/caching textures and compiling shaders. Subsequent launches are faster. Not a bug, just a cost of a nicer environment.

### Forward Movement

"Move forward" (two_fingers) commands `twist.linear.x` which moves the drone along its local X axis. This works, but the drone doesn't visually "tilt forward" like a real quadcopter. The Tello plugin's flight dynamics are simplified. This is a cosmetic limitation of the simulation, not a bug.

---

## File Reference

| File | Location | Purpose |
|------|----------|---------|
| `gesture_bridge.py` | `gesture_drone/scripts/` | Windows-side: webcam → MediaPipe → EfficientNet → TCP |
| `gesture_ros2_node.py` | `gesture_drone/scripts/` | WSL2-side: TCP → ROS2 topics/services → Gazebo |
| `launch_gazebo_bridge.sh` | `gesture_drone/scripts/` | One-command launcher for Gazebo + ROS2 node |
| `collect_data.py` | `gesture_drone/scripts/` | Webcam data collection tool |
| `crop_hands.py` | `gesture_drone/scripts/` | Batch hand cropping with MediaPipe |
| `train_model.py` | `gesture_drone/scripts/` | EfficientNet-B0 training pipeline |
| `gesture_model.pt` | `gesture_drone/models/` | Current trained model weights |
| `pretty.world` | `ros2_ws/.../tello_gazebo/worlds/` | Enhanced Gazebo world file |
| `pretty_launch.py` | `ros2_ws/.../tello_gazebo/launch/` | ROS2 launch file for pretty world |
| `SIMULATION_GUIDE.md` | project root | Full theory document for the simulation phase |
| `JOURNEY.md` | project root | Full theory document for the ML/perception phase |

---

## Bottom Line

The simulation infrastructure works. The robotics plumbing (ROS2, Gazebo, TCP bridge, state machine) is solid and doing its job. The weak link is the gesture classification model, and the root cause is insufficient training data diversity. More data from more people in more environments is the highest-impact next step. Everything else -- temporal filters, threshold tuning, augmentation -- is secondary.
