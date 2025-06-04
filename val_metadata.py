import os
import json
import cv2

ROOT_DIR = "lrs3"
OUTPUT_JSON = "val_metadata.json"

def find_metadata_files(root):
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith(".json"):
                yield os.path.join(dirpath, fname)

def extract_metadata(json_path):
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
        
        video_path = json_path.replace(".json", ".mp4")
        if not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        rel_path = os.path.relpath(video_path, ".")
        modify_type = meta.get("modify_type", "real")
        fake_segments = meta.get("fake_segments", [])

        return {
            "file": rel_path,
            "original": rel_path if modify_type == "real" else None,
            "split": "train",
            "modify_type": modify_type,
            "fake_segments": fake_segments,
            "video_frames": frame_count
        }
    except Exception as e:
        print(f"[ERROR] Failed to process {json_path}: {e}")
        return None

def main():
    all_samples = []
    for json_path in find_metadata_files(ROOT_DIR):
        sample = extract_metadata(json_path)
        if sample:
            all_samples.append(sample)

    with open(OUTPUT_JSON, "w") as out:
        json.dump(all_samples, out, indent=2)

    print(f"Saved {len(all_samples)} entries to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
