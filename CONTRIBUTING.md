# Contributing

Contributions are welcome for:

- Piper data collection reliability
- D405/D455 point-cloud preprocessing
- FP3 dataset conversion and audits
- inference latency
- action-chunk execution
- reproducibility documentation
- hardware-independent tests

Please avoid committing:

- real HDF5 datasets
- pretrained or fine-tuned weights
- SSH keys / tokens
- private calibration or personal information

Before opening a pull request:

```bash
python tools/public_release_check.py
python -m compileall -q windows remote tools tests
```
