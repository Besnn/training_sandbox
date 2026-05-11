import torch
from pathlib import Path
from training_fomo import FOMO_PL_480

MODEL_IN = "fomo_pl.pt"
MODEL_OUT = "models/fomo.onnx"

Path("models").mkdir(parents=True, exist_ok=True)

model = FOMO_PL_480(num_classes=4)
model.load_state_dict(torch.load(MODEL_IN, map_location="cpu"))
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