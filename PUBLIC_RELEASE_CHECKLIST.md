# GitHub public release checklist

## Repository creation

Use **New repository**, not **Import repository**.

Recommended:

- Owner: `sistaascencion996-hub`
- Repository name: `piper-fp3-pipeline`
- Description:
  `End-to-end AgileX Piper + D405/D455 + FP3 data collection, fine-tuning, inference and deployment pipeline.`
- Visibility: Public
- Initialize README: No
- Add .gitignore: No
- Choose license in GitHub UI: No

The repository package already contains README, .gitignore and LICENSE.

## Before first push

1. Confirm `config/pipeline.local.json` is ignored.
2. Run:
   `python tools/public_release_check.py`
3. Confirm no:
   - `.pth`
   - `.h5`
   - SSH keys
   - GitHub PAT
   - private dataset links
4. Review `THIRD_PARTY_NOTICES.md`.
5. Add a demo GIF/video later if desired.

## After first push

Recommended GitHub topics:

- robotics
- imitation-learning
- diffusion-policy
- fp3
- agilex
- piper
- realsense
- point-cloud
- robot-learning
- manipulation

Recommended first release tag:

`v0.1.0`
