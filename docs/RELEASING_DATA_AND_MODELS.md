# Model and dataset release strategy

GitHub should contain the **code chain**, not every byte of the data chain.

## Put in GitHub

- data collector
- converter
- audits
- training patch / config generator
- inference server
- robot client
- configuration examples
- dataset schema
- checksums / manifests
- evaluation results
- small screenshots or compressed demo media when useful

## Keep outside Git

- raw RGB-D recordings
- `.h5` episodes
- `.pth` checkpoints
- upstream FP3 pretrained weight
- private calibration files
- SSH credentials

For reproducibility, publish external download references plus SHA256 checksums.
