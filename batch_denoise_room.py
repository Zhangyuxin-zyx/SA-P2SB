import os
import argparse
import subprocess

def is_pointcloud_file(filename):
    return filename.endswith(".ply") or filename.endswith(".xyz")


def run_denoise(input_path, output_path, args):
    cmd = [
        "python", "denoise_room.py",
        "--room_path", input_path,
        "--model_path", args.model_path,
        "--batch_size", str(args.batch_size),
        "--steps", str(args.steps),
        "--k", str(args.k),
        "--out_path", output_path,
        "--overwrite"
    ]

    print(f"[INFO] Processing: {input_path}")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--model_path", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--k", type=int, default=4)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(os.listdir(args.input_dir))

    for file in files:
        if not is_pointcloud_file(file):
            continue

        input_path = os.path.join(args.input_dir, file)

        name = os.path.splitext(file)[0]
        output_path = os.path.join(args.output_dir, name + "_denoised.ply")

        if os.path.exists(output_path):
            print(f"[SKIP] Already exists: {output_path}")
            continue

        run_denoise(input_path, output_path, args)

    print("[DONE] All files processed.")


if __name__ == "__main__":
    main()