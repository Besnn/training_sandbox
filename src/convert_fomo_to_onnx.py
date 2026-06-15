import torch
from pathlib import Path
from training_fomo_v2 import FOMO_PL_480

MODEL_IN = "models/fomo-pt/fomo_v3_480.pt"
MODEL_OUT = "models/fomo-480-onnx/fomo-480-v3.onnx"

Path(MODEL_OUT).parent.mkdir(parents=True, exist_ok=True)

# Auto-detects the backbone width (alpha 1.0 / 0.35) from the checkpoint, so this
# works for both pre-alpha-0.35 (1.0) and new 0.35 FOMO models.
model = FOMO_PL_480.from_checkpoint(MODEL_IN, num_classes=4)
model.eval()

dummy_input = torch.randn(1, 3, 480, 480)

torch.onnx.export(
    model,
    dummy_input,
    MODEL_OUT,
    opset_version=21,
    export_params=True,
    external_data=False,
    input_names=["images"],
    output_names=["output"],
)

print(f"Esportazione completata in {MODEL_OUT}")