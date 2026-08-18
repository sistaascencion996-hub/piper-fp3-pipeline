# Dataset schema

## Raw episode

Canonical hardware:

- wrist camera: Intel RealSense D405
- external camera: Intel RealSense D455
- robot: AgileX Piper follower

The raw recorder stores synchronized RGB-D and robot feedback in `frames.h5`.

## Converted FP3 HDF5

Expected high-level structure:

```text
action/
  abs_pos                  [T, 3]
  abs_rot_6d               [T, 6]
  gripper_position         [T, 1]

observation/
  robot_state/
    cartesian_position     [T, 6]
    gripper_position       [T, 1]
  camera/
    pointcloud/
      hand_camera_left_pcd_4000
      varied_camera_2_left_pcd_4000

debug/
  ...
```

In this Piper adaptation each live camera stream is converted to a camera-local
XYZRGB cloud. The training pipeline currently uses 8000 points per camera per
frame even though the historical FP3 key names retain `_4000`.

## Action timing

The conversion uses:

`action[t] = follower_state[t+1]`

with the final action padded by the final state.

## Rotation representation

The Piper converter uses FP3's 6D rotation representation from the first two
**rows** of the 3x3 rotation matrix. The robot client reconstructs the rows,
orthonormalizes them, and converts back to Piper RPY.

This row/column convention is important; treating the 6D vectors as matrix
columns can produce invalid Cartesian targets and joint-limit errors.
