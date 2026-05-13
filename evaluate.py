import os
import argparse
import torch
import numpy as np
import pandas as pd
import open3d as o3d
from tqdm import tqdm

from metrics.metrics import cd_unit_sphere, point_face_dist

def main():
    parser = argparse.ArgumentParser(description="Point Cloud Denoising Evaluation Script")
    parser.add_argument("--pred_dir", type=str, required=True, help="Denoised point cloud (.ply or .xyz) directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="The directory where the GT point cloud is located")

    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the point cloud to the unit sphere before calculation")
    args = parser.parse_args()

    MULTIPLIER = 10**3
    results = []

    all_pred_files = [f for f in os.listdir(args.pred_dir) if f.endswith(".ply") or f.endswith(".xyz")]

    if not all_pred_files:
        print(f"No .ply or .xyz files found in {args.pred_dir}!")
        return

    print(f"Find {len(all_pred_files)} denoised point cloud files and start evaluation (Normalize: {args.normalize})...")

    for pred_file in tqdm(all_pred_files, desc="Processing"):
        base_name, ext = os.path.splitext(pred_file)
        
        if base_name.endswith("_denoised"):
            core_name = base_name[:-9] 
        else:
            core_name = base_name

        gt_file = f"{core_name}.ply"
        
        pred_path = os.path.join(args.pred_dir, pred_file)
        gt_path = os.path.join(args.gt_dir, gt_file)

        if not os.path.exists(gt_path):
            print(f"\n Warning: The GT file ({gt_path}) corresponding to {pred_file} cannot be found, skipping...")
            continue

        pred_pcd = o3d.io.read_point_cloud(pred_path)
        pred_pts = torch.from_numpy(np.array(pred_pcd.points)).float().cuda()
        if pred_pts.ndim == 2:
            pred_pts = pred_pts.unsqueeze(0)

        gt_pcd = o3d.io.read_point_cloud(gt_path)
        gt_pts = torch.from_numpy(np.array(gt_pcd.points)).float().cuda()
        gt_mesh = o3d.io.read_triangle_mesh(gt_path)

        if gt_pts.ndim == 2:
            gt_pts = gt_pts.unsqueeze(0)

        cd_acc, cd_comp = cd_unit_sphere(pred_pts, gt_pts, normalize=args.normalize)
        cd_acc *= MULTIPLIER
        cd_comp *= MULTIPLIER
        cd_total = (cd_acc + cd_comp) / 2

        p2f, f2p, p2m = np.nan, np.nan, np.nan
        
        if not gt_mesh.has_triangles():
            print(f"\n Warning: GT file {gt_file} does not contain triangle faces (Faces)! P2F/F2P/P2M are recorded as NaN.")
        else:
            gt_verts = torch.tensor(np.array(gt_mesh.vertices)).float().cuda()
            gt_faces = torch.tensor(np.array(gt_mesh.triangles)).long().cuda()
            
            point_dist, face_dist = point_face_dist(
                pred_pts.squeeze(0), gt_verts, gt_faces, normalize=args.normalize
            )
            
            p2f = point_dist * MULTIPLIER
            f2p = face_dist * MULTIPLIER
            p2m = p2f  # Mathematically equivalent

        results.append({
            "File": pred_file, 
            "CD->": cd_acc,
            "CD<-": cd_comp,
            "CD": cd_total,
            "P2F": p2f,
            "F2P": f2p,
            "P2M": p2m
        })

    if not results:
        print("No files were successfully evaluated.")
        return

    df = pd.DataFrame(results)
    
    mean_row = {"File": "AVERAGE"}
    for col in ["CD->", "CD<-", "CD", "P2F", "F2P", "P2M"]:
        mean_row[col] = df[col].mean(skipna=True)
    
    df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)

    print("\n" + "="*85)
    print("Summary of evaluation results (unit: enlarged by MULTIPLIER)")
    print("="*85)
    print(df.to_markdown(index=False, floatfmt=".4f"))
    print("="*85)

    output_csv = "evaluation_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n Results saved to: ./{output_csv}")

if __name__ == "__main__":
    main()