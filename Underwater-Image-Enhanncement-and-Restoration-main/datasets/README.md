# Datasets

Place datasets under this folder. Each dataset should have:

- input/  # degraded underwater images (blurred/dark)
- target/ # corresponding references or enhanced images (optional for no-ref testing)

Example:

```
Underwater-Image-Enhanncement-and-Restoration-main/datasets/YourDataset/
  input/
    img1.jpg
    ...
  target/
    img1.png
    ...
```

For no-reference evaluation, you may leave `target/` empty and use UIQM/UCIQE only.
