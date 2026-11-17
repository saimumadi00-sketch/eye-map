# Object Directions and Layout

EyeMap can describe what is around the camera in three ways, and draw the result as a top-down layout.

| What | Needs | Output |
| --- | --- | --- |
| Map clusters: groups of sparse map points (obstacles, structures) with direction and distance | nothing extra | `objects/scene.json`, `objects/clusters.csv` |
| Detected objects: label, direction, and distance where the map allows | an ONNX detection model | `objects/objects.csv`, `objects/scene.json`, annotated frames |
| Object motion: how each tracked object moves | an ONNX detection model | same files |
| Layout: top-down picture of path, camera, map, clusters and objects | nothing extra | `plots/layout.png`, live with `--show-layout` |

## Run It

The layout and map clusters are written for every run that triangulates points:

```bash
python main.py --source video --path flight.mp4 --no-display
```

Add a detector for labelled objects and motion:

```bash
python main.py --source video --path flight.mp4 --detect-model models/yolo11n.onnx --show-layout
```

Options (also settable in `config/config.yaml`):

- `--detect-model`: YOLO-style ONNX file. EyeMap ships no model; export or download one yourself and check its licence.
- `--detect-classes`: text file with one class name per line, in the model's order. Defaults to the 80 COCO classes.
- `--detect-conf`: confidence threshold (default 0.35).
- `--detect-interval N`: run the detector every Nth frame. Detection is the slowest stage, so this is the main speed control.
- `--show-layout`: live top-down window.
- `export_layout: false` in the config turns the exports off.

Supported output layouts are `(1, 4 + classes, N)` (YOLOv8, YOLO11) and `(1, N, 5 + classes)` (YOLOv5), read through OpenCV's DNN module with 640x640 letterboxed input.

## How to Read the Directions

All angles are relative to the camera's own axes at that moment.

- `azimuth_deg`: degrees to the right of straight ahead (negative is left).
- `elevation_deg`: degrees above straight ahead (negative is below).
- `direction`: `ahead` (within 5 degrees), `slightly left/right` (to 22.5), `ahead-left/right` (to 67.5), `left/right`, `behind-left/right`, `behind`.
- `elevation`: `above`, `level`, or `below` (10-degree band).
- `distance`: straight-line distance in map units. With `--gps-csv`, `scene.json` also gives `distance_m`.

In `objects.csv` each row is one object in one frame, seen from that frame's camera. In `scene.json`, `last_seen` is the direction when the object was last detected, and `from_final_camera` is where it lies from the camera's final position.

## How Distance Is Found

A detector gives a box, not a depth. EyeMap projects recent sparse map points into the frame and uses those that fall inside the middle of the box, taking the nearer quartile so background at the edges counts less. An object gets a distance only if at least 3 map points land on it; otherwise `distance` is empty and only the direction is reported.

## How Motion Is Reported

- `apparent_motion` (`left`, `right`, `up`, `down`, `still`): movement across the image. It includes the camera's own motion: a parked car drifts left when the camera moves right.
- `range_trend` (`approaching`, `receding`, `steady`): from the box growing or shrinking.
- `world_motion` (`toward`, `away`, `left`, `right`, `stationary`): movement in the map, relative to the camera's view, so it is independent of camera motion. Only available when the object has a distance.

## Limits

- Distances are in relative map units unless GPS fixes supply the scale.
- A moving object's own feature points do not triangulate, so its distance usually comes from the background behind it. Its `world_motion` is then approximate; trust the direction of motion more than the speed.
- `world_motion` needs motion of at least 1% of the object's distance per frame to register; slower movement reads as `stationary`.
- Tracking matches boxes of the same class by overlap. Objects that cross or leave for more than 15 frames get new ids.
- Map clusters come from grouping points on the ground plane (x-z). They are geometry only: a cluster may be one object, several, or part of the ground.
- The layout looks down the world y axis, which is "down" only if the first image was taken with the camera level.
- Tested with a YOLO11n COCO model on a photo and a synthetic video. It has not been evaluated on real drone footage.
