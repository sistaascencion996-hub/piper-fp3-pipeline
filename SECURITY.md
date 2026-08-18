# Security

## Reporting

Please do not publish robot-network credentials, SSH keys, personal access
tokens, private IP routing details, or private dataset links in GitHub issues.

## Secrets

The repository intentionally ignores:

- `.env`
- `config/pipeline.local.json`
- SSH keys
- model checkpoints
- HDF5 datasets
- videos
- raw camera recordings

Run `python tools/public_release_check.py` before every public push.

## Physical robot safety

This repository can command a physical robot. Test changes in an appropriate
controlled environment and keep an independent hardware stop mechanism
available. Repository maintainers cannot validate a user's hardware setup.
