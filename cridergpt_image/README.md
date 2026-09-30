# CriderGPT Image

CriderGPT Image is the native image-generation component planned for the CriderGPT engine.

## Stage 1 status

Stage 1 defines the product and technical contract only. It does **not** claim that an image model has been trained yet.

The target is an independently trained CriderGPT image generator whose inference path does not call OpenAI, Gemini, Midjourney, Stable Diffusion services, or another hosted AI model.

See `stage1_spec.json` for the machine-readable Stage 1 contract.

## Stage 1 exit criteria

Stage 1 is complete when the initial generation target, independence requirements, output contract, reproducibility requirements, and later-stage boundaries are fixed. Stage 2 can then build the image/caption dataset pipeline against this contract.
