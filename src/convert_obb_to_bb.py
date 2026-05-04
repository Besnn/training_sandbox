import os
import glob

INPUT_DIR = "datasets/obb_dataset/labels"

OUTPUT_DIR = "datasets/bb_dataset/labels"

os.makedirs(OUTPUT_DIR, exist_ok=True)

txt_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))

if not txt_files:
    print(f"No .txt files found in '{INPUT_DIR}'. Please check the path.")

processed_count = 0

for txt_file in txt_files:
    with open(txt_file, "r") as file:
        lines = file.readlines()

    new_lines = []

    for line in lines:
        parts = line.strip().split()

        if len(parts) >= 9:
            class_id = parts[0]

            xs = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
            ys = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]

            xmin = min(xs)
            xmax = max(xs)
            ymin = min(ys)
            ymax = max(ys)

            x_center = min(max((xmin + xmax) / 2.0, 0.0), 1.0)
            y_center = min(max((ymin + ymax) / 2.0, 0.0), 1.0)
            width = min(max(xmax - xmin, 0.0), 1.0)
            height = min(max(ymax - ymin, 0.0), 1.0)

            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        else:
            new_lines.append(line)

    filename = os.path.basename(txt_file)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(output_path, "w") as file:
        file.writelines(new_lines)

    processed_count += 1

print(f"Successfully converted {processed_count} files from OBB to standard BB format!")
print(f"Your new labels are in: '{OUTPUT_DIR}'")