import torch
from pathlib import Path

orig = Path("models/tinytron/Qwen2-7B-Instruct-Tinytron/pytorch_model.bin")
new  = Path("models/tinytron/Qwen2-7B-Instruct-Tinytron/pytorch_model_new.bin")

print(f"Loading original checkpoint from: {orig}")
state = torch.load(orig, map_location="cpu", weights_only=False)  # old format OK

print("Saving checkpoint with new zipfile serialization...")
torch.save(state, new, _use_new_zipfile_serialization=True)

print(f"Saved new-format checkpoint to: {new}")

