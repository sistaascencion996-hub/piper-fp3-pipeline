# Model artifacts

Do not commit `.pth`, `.ckpt`, `.pt`, `.safetensors` or other large model files.

The current pipeline selects:

`model_best_train_loss.pth`

using the globally lowest epoch training loss requested by this project.

Recommended public release pattern:

- GitHub repository: code + configs + docs
- model hosting: a dedicated model host / institutional storage
- GitHub Release notes: model metadata + SHA256 + download location

The upstream FP3 pretrained checkpoint is not redistributed here.
