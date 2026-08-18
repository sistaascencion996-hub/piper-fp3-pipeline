# End-to-end pipeline

## 1. Collect

Windows:

```powershell
.\pipeline.ps1 collect -Episode 21 -Duration 30
```

## 2. Upload

```powershell
.\pipeline.ps1 upload -Episode 21
```

## 3. Convert + audit

RTX host:

```bash
./pipeline_remote.sh prepare
```

## 4. Train

```bash
./pipeline_remote.sh train
```

Training policy used by this repository:

- 2000 epochs
- 5 optimizer steps per epoch
- 10000 optimizer steps total
- learning rate 1e-4
- EMA enabled
- one retained checkpoint: historical minimum epoch train loss

## 5. Serve

```bash
./pipeline_remote.sh serve
```

The wrapper intentionally does not pass `--replan-every-step`.

## 6. Run

Windows:

```powershell
.\pipeline.ps1 run
```

The current robot client:

- uses Piper firmware family V189
- reconstructs FP3 rot6d with the correct row convention
- records startup joints
- on Ctrl+C stops policy commands and returns to the startup joint pose
