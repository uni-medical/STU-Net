#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Visualization for Head & Neck Cancer Segmentation with Anatomical Background
Creates interactive HTML and multi-view images with H&N anatomy context

Uses TotalSegmentator output for anatomical structures as background

Outputs:
    - 3D interactive HTML visualization
    - Multi-view PNG images (6 rotation angles)
    - Combined visualization with tumors and anatomy
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
import pyvista as pv
from pathlib import Path
from skimage import measure

# Fix numpy compatibility for older skimage (NumPy 2.0+ compatibility)
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float

# Start Xvfb for headless rendering
try:
    pv.start_xvfb()
except:
    pass


# TotalSegmentator labels relevant for Head & Neck visualization
HN_ANATOMY_LABELS = {
    # Bones/Skeleton (show as light gray, transparent)
    'vertebrae_C1': {'label': 81, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C2': {'label': 82, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C3': {'label': 83, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C4': {'label': 84, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C5': {'label': 85, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C6': {'label': 86, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_C7': {'label': 87, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_T1': {'label': 93, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_T2': {'label': 97, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    'vertebrae_T3': {'label': 98, 'color': '#E8E8E8', 'opacity': 0.15, 'group': 'spine'},
    
    # Shoulders/clavicles
    'clavicula_left': {'label': 7, 'color': '#D0D0D0', 'opacity': 0.15, 'group': 'bones'},
    'clavicula_right': {'label': 8, 'color': '#D0D0D0', 'opacity': 0.15, 'group': 'bones'},
    'scapula_left': {'label': 74, 'color': '#C8C8C8', 'opacity': 0.12, 'group': 'bones'},
    'scapula_right': {'label': 75, 'color': '#C8C8C8', 'opacity': 0.12, 'group': 'bones'},
    'humerus_left': {'label': 29, 'color': '#D8D8D8', 'opacity': 0.10, 'group': 'bones'},
    'humerus_right': {'label': 30, 'color': '#D8D8D8', 'opacity': 0.10, 'group': 'bones'},
    
    # Ribs (top ones)
    'rib_left_1': {'label': 49, 'color': '#D4D4D4', 'opacity': 0.12, 'group': 'ribs'},
    'rib_right_1': {'label': 61, 'color': '#D4D4D4', 'opacity': 0.12, 'group': 'ribs'},
    'rib_left_2': {'label': 53, 'color': '#D4D4D4', 'opacity': 0.12, 'group': 'ribs'},
    'rib_right_2': {'label': 65, 'color': '#D4D4D4', 'opacity': 0.12, 'group': 'ribs'},
    
    # Face/brain for context
    'face': {'label': 12, 'color': '#FFE4C4', 'opacity': 0.08, 'group': 'head'},
    'brain': {'label': 6, 'color': '#FFB6C1', 'opacity': 0.10, 'group': 'head'},
    
    # Soft tissue structures
    'esophagus': {'label': 11, 'color': '#FFA07A', 'opacity': 0.20, 'group': 'soft'},
    'trachea': {'label': 79, 'color': '#87CEEB', 'opacity': 0.20, 'group': 'soft'},
    
    # Lungs (upper lobes for context)
    'lung_upper_lobe_left': {'label': 44, 'color': '#ADD8E6', 'opacity': 0.08, 'group': 'lungs'},
    'lung_upper_lobe_right': {'label': 45, 'color': '#ADD8E6', 'opacity': 0.08, 'group': 'lungs'},
    
    # Vessels (aorta, for spatial reference)
    'aorta': {'label': 3, 'color': '#CD5C5C', 'opacity': 0.15, 'group': 'vessels'},
}

# Tumor labels
TUMOR_LABELS = {
    1: {'name': 'GTVp (Primary Tumor)', 'color': '#FF0000', 'opacity': 1.0},  # Bright red
    2: {'name': 'GTVn (Nodal)', 'color': '#00FF00', 'opacity': 1.0}  # Bright green
}


def create_mesh_from_label(segmentation_data, label_value, spacing=(1.0, 1.0, 1.0), smooth_iter=30):
    """Create a 3D mesh from a segmentation label using marching cubes."""
    
    binary_mask = (segmentation_data == label_value).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return None
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask, 
            level=0.5,
            spacing=spacing
        )
        
        faces_pv = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).flatten()
        
        mesh = pv.PolyData(verts, faces_pv)
        if smooth_iter > 0:
            mesh = mesh.smooth(n_iter=smooth_iter)
        
        return mesh
    except Exception as e:
        return None


def load_anatomy_meshes(anatomy_path, spacing):
    """Load anatomical structures from TotalSegmentator output."""
    
    print(f"\nLoading anatomical structures from: {anatomy_path}")
    
    anatomy_nii = nib.load(anatomy_path)
    anatomy_data = anatomy_nii.get_fdata().astype(np.uint8)
    
    unique_labels = np.unique(anatomy_data)
    print(f"  Found {len(unique_labels)} unique labels in anatomy file")
    
    anatomy_meshes = {}
    
    for struct_name, struct_info in HN_ANATOMY_LABELS.items():
        label_val = struct_info['label']
        if label_val in unique_labels:
            mesh = create_mesh_from_label(anatomy_data, label_val, spacing, smooth_iter=20)
            if mesh is not None:
                anatomy_meshes[struct_name] = {
                    'mesh': mesh,
                    'color': struct_info['color'],
                    'opacity': struct_info['opacity'],
                    'group': struct_info['group']
                }
                print(f"    ✓ {struct_name}: {mesh.n_points} points")
    
    print(f"  Loaded {len(anatomy_meshes)} anatomical structures")
    return anatomy_meshes


def visualize_hn_with_anatomy(
    segmentation_path, 
    output_dir, 
    anatomy_path=None,
    ct_path=None, 
    patient_id=None,
    show_anatomy=True
):
    """
    Create 3D visualization of Head & Neck cancer segmentation with anatomical background.
    
    Args:
        segmentation_path: Path to tumor segmentation NIfTI file
        output_dir: Output directory for visualizations
        anatomy_path: Path to TotalSegmentator output for anatomy
        ct_path: Optional path to CT (used for TotalSegmentator if anatomy_path not provided)
        patient_id: Patient identifier for labeling
        show_anatomy: Whether to show anatomical structures
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load tumor segmentation
    print(f"\nLoading tumor segmentation: {segmentation_path}")
    seg_nii = nib.load(segmentation_path)
    seg_data = seg_nii.get_fdata().astype(np.uint8)
    spacing = seg_nii.header.get_zooms()[:3]
    
    print(f"  Shape: {seg_data.shape}")
    print(f"  Spacing: {spacing}")
    
    if patient_id is None:
        patient_id = Path(segmentation_path).stem.replace('.nii', '')
    
    # Create tumor meshes
    unique_labels = np.unique(seg_data).astype(int)
    print(f"  Tumor labels found: {unique_labels}")
    
    tumor_meshes = {}
    for label_val, label_info in TUMOR_LABELS.items():
        if label_val in unique_labels:
            print(f"  Creating mesh for {label_info['name']}...")
            mesh = create_mesh_from_label(seg_data, label_val, spacing, smooth_iter=50)
            if mesh is not None:
                tumor_meshes[label_val] = mesh
                voxel_count = np.sum(seg_data == label_val)
                print(f"    Mesh: {mesh.n_points} points, {mesh.n_cells} faces ({voxel_count:,} voxels)")
    
    if not tumor_meshes:
        print("  ERROR: No tumor meshes could be created!")
        return None
    
    # Load anatomy if available
    anatomy_meshes = {}
    if show_anatomy and anatomy_path and os.path.exists(anatomy_path):
        anatomy_meshes = load_anatomy_meshes(anatomy_path, spacing)
    
    # ============================================================
    # 1. Create Interactive HTML Visualization
    # ============================================================
    print(f"\nCreating interactive 3D HTML visualization...")
    
    html_path = os.path.join(output_dir, f'{patient_id}_3d_with_anatomy.html')
    
    try:
        plotter_html = pv.Plotter(off_screen=True)
        
        # Add anatomy first (background, lower opacity)
        for struct_name, struct_info in anatomy_meshes.items():
            plotter_html.add_mesh(
                struct_info['mesh'], 
                color=struct_info['color'], 
                opacity=struct_info['opacity'],
                smooth_shading=True
            )
        
        # Add tumors on top (foreground, full opacity)
        for label_val, mesh in tumor_meshes.items():
            label_info = TUMOR_LABELS[label_val]
            plotter_html.add_mesh(
                mesh, 
                color=label_info['color'], 
                opacity=label_info['opacity'],
                label=label_info['name'],
                smooth_shading=True
            )
        
        plotter_html.add_legend()
        plotter_html.set_background('#1a1a2e')
        plotter_html.add_axes()
        
        plotter_html.export_html(html_path)
        print(f"  Saved: {html_path}")
        plotter_html.close()
    except Exception as e:
        print(f"  Warning: Could not create HTML: {e}")
    
    # ============================================================
    # 2. Create Multi-View PNG Images (6 angles)
    # ============================================================
    print(f"\nCreating multi-view images with anatomy background...")
    
    view_angles = [0, 60, 120, 180, 240, 300]
    view_images = []
    
    # Calculate bounds from tumors for centering
    all_tumor_points = np.vstack([mesh.points for mesh in tumor_meshes.values()])
    tumor_center = all_tumor_points.mean(axis=0)
    
    for angle in view_angles:
        plotter = pv.Plotter(off_screen=True, window_size=[600, 800])
        
        # Add anatomy structures as background
        for struct_name, struct_info in anatomy_meshes.items():
            plotter.add_mesh(
                struct_info['mesh'], 
                color=struct_info['color'], 
                opacity=struct_info['opacity'],
                smooth_shading=True
            )
        
        # Add tumor meshes in foreground
        for label_val, mesh in tumor_meshes.items():
            label_info = TUMOR_LABELS[label_val]
            plotter.add_mesh(
                mesh, 
                color=label_info['color'], 
                opacity=label_info['opacity'],
                smooth_shading=True
            )
        
        # Add coordinate axes
        bounds = plotter.bounds
        x_len = bounds[1] - bounds[0]
        y_len = bounds[3] - bounds[2]
        z_len = bounds[5] - bounds[4]
        
        axis_scale = 0.2
        axis_origin = np.array([
            bounds[0] - x_len * 0.1,
            bounds[2] - y_len * 0.1,
            bounds[4] - z_len * 0.05
        ])
        
        axis_colors = ['#0066FF', '#FF9900', '#00CC00']
        axis_labels = ['X', 'Y', 'Z']
        axis_lengths = [x_len * axis_scale, y_len * axis_scale, z_len * axis_scale]
        
        for i, (ax_len, ax_color, ax_label) in enumerate(zip(axis_lengths, axis_colors, axis_labels)):
            direction = np.zeros(3)
            direction[i] = 1.0
            
            arrow = pv.Arrow(
                start=axis_origin,
                direction=direction,
                scale=ax_len,
                shaft_radius=0.02,
                tip_radius=0.05,
                tip_length=0.2
            )
            plotter.add_mesh(arrow, color=ax_color)
            
            label_pos = axis_origin + direction * ax_len * 1.15
            plotter.add_point_labels(
                [label_pos], 
                [f'{ax_label}\n{ax_len:.0f}mm'],
                font_size=12,
                text_color=ax_color,
                shape=None,
                fill_shape=False
            )
        
        # Camera setup
        plotter.camera_position = 'xz'
        plotter.camera.azimuth = angle
        plotter.camera.elevation = 15
        plotter.reset_camera()
        plotter.camera.zoom(0.75)  # Zoom out a bit to show anatomy
        
        plotter.set_background('#FFFFFF')
        
        view_path = os.path.join(output_dir, f'{patient_id}_anatomy_view_{angle:03d}.png')
        plotter.screenshot(view_path)
        view_images.append(view_path)
        print(f"    Saved view {angle}°")
        plotter.close()
    
    # ============================================================
    # 3. Create Combined Multi-View Image (2x3 grid)
    # ============================================================
    print(f"\nCreating combined multi-view image...")
    
    combined_path = None
    try:
        from PIL import Image, ImageDraw, ImageFont
        
        images = [Image.open(p) for p in view_images]
        
        img_width, img_height = images[0].size
        grid_width = img_width * 3
        grid_height = img_height * 2 + 60  # Extra space for legend
        
        combined = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for idx, img in enumerate(images):
            row = idx // 3
            col = idx % 3
            combined.paste(img, (col * img_width, row * img_height))
        
        # Add legend at bottom
        draw = ImageDraw.Draw(combined)
        legend_y = img_height * 2 + 10
        
        # Legend items
        legend_items = [
            ('GTVp (Primary Tumor)', '#FF0000'),
            ('GTVn (Nodal Metastases)', '#00FF00'),
            ('Vertebrae/Spine', '#E8E8E8'),
            ('Trachea', '#87CEEB'),
            ('Esophagus', '#FFA07A'),
        ]
        
        x_pos = 20
        for label_name, color in legend_items:
            # Draw color box
            draw.rectangle([x_pos, legend_y, x_pos + 20, legend_y + 20], fill=color, outline='black')
            # Draw label
            draw.text((x_pos + 25, legend_y + 2), label_name, fill='black')
            x_pos += len(label_name) * 8 + 50
        
        combined_path = os.path.join(output_dir, f'{patient_id}_anatomy_multiview.png')
        combined.save(combined_path, dpi=(150, 150))
        print(f"  Saved: {combined_path}")
        
    except ImportError:
        print("  Warning: PIL not available, skipping combined image")
    
    # ============================================================
    # 4. Save VTK meshes
    # ============================================================
    print(f"\nSaving VTK meshes...")
    
    for label_val, mesh in tumor_meshes.items():
        label_info = TUMOR_LABELS[label_val]
        vtk_path = os.path.join(output_dir, f'{patient_id}_tumor_label{label_val}.vtk')
        mesh.save(vtk_path)
        print(f"  Saved: {vtk_path}")
    
    # Save anatomy meshes too
    for struct_name, struct_info in anatomy_meshes.items():
        vtk_path = os.path.join(output_dir, f'{patient_id}_anatomy_{struct_name}.vtk')
        struct_info['mesh'].save(vtk_path)
    
    # ============================================================
    # 5. Print Summary
    # ============================================================
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nPatient: {patient_id}")
    print(f"Output directory: {output_dir}")
    print(f"\nTumors visualized:")
    for label_val, label_info in TUMOR_LABELS.items():
        if label_val in tumor_meshes:
            print(f"  - {label_info['name']} ({label_info['color']})")
    
    print(f"\nAnatomy structures shown: {len(anatomy_meshes)}")
    for group in ['spine', 'bones', 'soft', 'head', 'lungs', 'vessels']:
        structs = [k for k, v in anatomy_meshes.items() if v['group'] == group]
        if structs:
            print(f"  - {group}: {', '.join(structs)}")
    
    print(f"\nFiles created:")
    print(f"  - {patient_id}_3d_with_anatomy.html")
    print(f"  - {patient_id}_anatomy_multiview.png")
    print(f"  - Individual view PNGs and VTK meshes")
    
    return {
        'html': html_path,
        'multiview': combined_path,
        'views': view_images,
        'tumor_meshes': tumor_meshes,
        'anatomy_meshes': anatomy_meshes
    }


def main():
    parser = argparse.ArgumentParser(
        description='3D Visualization for H&N Cancer with Anatomical Background',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # With pre-computed anatomy
  python visualize_hn_with_anatomy.py -i output_data/CHUM-001/CHUM-001.nii.gz \\
      -o viz/CHUM-001 --anatomy anatomy_data/CHUM-001.nii
  
  # Run TotalSegmentator first if anatomy not available:
  TotalSegmentator -i CT.nii.gz -o anatomy/ --ml --fast
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to tumor segmentation NIfTI file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--anatomy', type=str, default=None,
                        help='Path to TotalSegmentator anatomy segmentation')
    parser.add_argument('--patient-id', type=str, default=None,
                        help='Patient ID for labeling (default: from filename)')
    parser.add_argument('--no-anatomy', action='store_true',
                        help='Disable anatomical background')
    
    args = parser.parse_args()
    
    visualize_hn_with_anatomy(
        segmentation_path=args.input,
        output_dir=args.output,
        anatomy_path=args.anatomy,
        patient_id=args.patient_id,
        show_anatomy=not args.no_anatomy
    )


if __name__ == '__main__':
    main()
