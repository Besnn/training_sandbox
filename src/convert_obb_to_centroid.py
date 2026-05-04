import os
import glob

LABEL_DIR = "datasets/centroid_dataset/labels"

for txt_file in glob.glob(os.path.join(LABEL_DIR, "*.txt")):
    with open(txt_file, "r") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 9:
            class_id = parts[0]
            xs = [float(parts[1]), float(parts[3]), float(parts[5]), float(parts[7])]
            ys = [float(parts[2]), float(parts[4]), float(parts[6]), float(parts[8])]

            x_center = sum(xs) / 4.0
            y_center = sum(ys) / 4.0

            # Diamo una larghezza e altezza finte (es. 0.05)
            # perché i framework richiedono i 4 valori, anche se FOMO userà solo il centro
            fake_width = 0.05
            fake_height = 0.05

            new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {fake_width} {fake_height}\n")

    with open(txt_file, "w") as file:
        file.writelines(new_lines)

print("Conversione OBB -> FOMO completata")