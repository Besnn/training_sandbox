import splitfolders

input_folder = "datasets/obb_dataset"
output_folder = "datasets/split_obb_dataset"

# Dividi in: 80% Training, 10% Validation, 10% Test
splitfolders.ratio(input_folder, output=output_folder,
                   seed=42, ratio=(.8, .1, .1),
                   group_prefix=None, move=False)

print("Split completato.")