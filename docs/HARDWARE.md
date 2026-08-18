# Hardware profile used during development

Reference configuration:

- AgileX Piper follower arm
- Intel RealSense D405 wrist camera
- Intel RealSense D455 external camera
- Windows robot/control computer
- remote NVIDIA RTX 4090 training/inference host
- CAN backend: `agx_cando`
- Piper firmware family used by the client: `V189`

All serial numbers, IP addresses, user names and absolute paths belong in
`config/pipeline.local.json`, which is ignored by Git.
