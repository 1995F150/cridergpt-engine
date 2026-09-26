# CriderGPT Model Runtime

This directory tracks the source/configuration needed to deploy the CriderGPT model runtime.

The repository currently contains a smoke-test inference backend, not trained model weights. It verifies that the server can pull and execute the model integration before a real checkpoint is connected.

## Server test

```bash
python3 cridergpt_stage5/inference.py "Hello CriderGPT"
```

Expected output includes `CriderGPT model runtime received: Hello CriderGPT`.

Large trained model weights should be deployed separately rather than committed directly to normal Git history.
