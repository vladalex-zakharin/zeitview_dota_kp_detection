import os
import json

def parse_label_file(label_path):
    keypoints = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) < 9:
                continue  # skip bad rows
            coords = list(map(float, parts[:8]))
            x_coords = coords[0::2]
            y_coords = coords[1::2]
            x_center = sum(x_coords) / 4.0
            y_center = sum(y_coords) / 4.0
            keypoints.append([x_center, y_center])
    return keypoints


def create_dataset(images_dir, labels_dir, output_json):
    data = []
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]

    for img_file in image_files:
        img_id = img_file.split('.')[0]
        label_path = os.path.join(labels_dir, f"{img_id}.txt")
        if not os.path.exists(label_path):
            continue

        keypoints = parse_label_file(label_path)

        data.append({
            "image_file": img_file,
            "keypoints": keypoints
        })
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved processed dataset to {output_json}")


if __name__ == "__main__":
    create_dataset(
        images_dir="data/raw/train/images",
        labels_dir="data/raw/train/labelTxt",
        output_json="data/processed/train_keypoints.json"
    )
