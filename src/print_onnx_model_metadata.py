import onnx
import sys
import os


def print_onnx_info(model_path):
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} not found.")
        return

    # Load the model
    model = onnx.load(model_path)

    print(f"--- Model Information: {os.path.basename(model_path)} ---")

    # Internal Model Name (Producer Name)
    print(f"Producer Name: {model.producer_name}")
    print(f"Producer Version: {model.producer_version}")

    # Domain and Model Version
    print(f"Domain: {model.domain}")
    print(f"Model Version: {model.model_version}")

    # Doc String (Often where specific model names/descriptions are stored)
    if model.doc_string:
        print(f"Description: {model.doc_string}")
    else:
        print("Description: No internal description provided.")

    # IR Version
    print(f"ONNX IR Version: {model.ir_version}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_model.onnx>")
    else:
        print_onnx_info(sys.argv[1])