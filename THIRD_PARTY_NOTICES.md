# Third-party notices

This repository is an integration layer. It intentionally does **not** vendor
the FP3 source tree, pretrained weights, pyAgxArm source, Intel RealSense SDK,
or demonstration datasets.

## FP3 / 3D Foundation Policy

Upstream project:

- Repository: `horipse01/3d-foundation-policy`
- Project: FP3: A 3D Foundation Policy for Robotic Manipulation
- Authors: Rujia Yang, Geng Chen, Chuan Wen, Yang Gao

The upstream GitHub repository currently does not advertise a repository
license in its GitHub metadata. For that reason this project does not copy the
FP3 source tree or pretrained weights into this repository. Users must obtain
FP3 separately and review the upstream terms themselves.

Our MIT license does not apply to FP3.

## pyAgxArm

This project imports `pyAgxArm` as an external dependency. The pyAgxArm project
is maintained by AgileX Robotics and is licensed separately (LGPL-3.0 at the
time this repository was prepared).

Our MIT license does not apply to pyAgxArm.

## Intel RealSense

The project uses `pyrealsense2` as an external dependency. Intel RealSense
software remains governed by its own license.

## Scientific Python dependencies

NumPy, SciPy, h5py, OpenCV, PyTorch and other dependencies remain governed by
their respective licenses.

No third-party dependency is relicensed by this repository.
