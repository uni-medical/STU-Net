#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
3D Visualization for Head & Neck Cancer Segmentation
Creates interactive HTML and multi-view images similar to AortaPlot

Outputs:
    - 3D interactive HTML visualization
    - Multi-view PNG images (6 rotation angles)
    - Overlay visualization with CT
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


def create_mesh_from_label(segmentation_data, label_value, spacing=(1.0, 1.0, 1.0)):
    """Create a 3D mesh from a segmentation label using marching cubes."""
    
    # Extract binary mask for this label
    binary_mask = (segmentation_data == label_value).astype(np.uint8)
    
    if binary_mask.sum() == 0:
        return None
    
    # Apply marching cubes to create surface mesh
    try:
        verts, faces, normals, values = measure.marching_cubes(
            binary_mask, 
            level=0.5,
            spacing=spacing
        )
        
        # Create PyVista mesh
        # Faces need to be in the format: [n_points, p1, p2, p3, ...]
        faces_pv = np.column_stack([
            np.full(len(faces), 3),
            faces
        ]).flatten()
        
        mesh = pv.PolyData(verts, faces_pv)
        mesh = mesh.smooth(n_iter=50)  # Smooth the mesh
        
        return mesh
    except Exception as e:
        print(f"  Warning: Could not create mesh for label {label_value}: {e}")
        return None


def visualize_hn_segmentation(segmentation_path, output_dir, ct_path=None, patient_id=None):
    """
    Create 3D visualization of Head & Neck cancer segmentation.
    
    Args:
        segmentation_path: Path to segmentation NIfTI file
        output_dir: Output directory for visualizations
        ct_path: Optional path to CT for overlay (not used in mesh, just for reference)
        patient_id: Patient identifier for labeling
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load segmentation
    print(f"\nLoading segmentation: {segmentation_path}")
    seg_nii = nib.load(segmentation_path)
    seg_data = seg_nii.get_fdata().astype(np.uint8)
    
    # Get spacing from header
    spacing = seg_nii.header.get_zooms()[:3]
    print(f"  Shape: {seg_data.shape}")
    print(f"  Spacing: {spacing}")
    
    # Extract patient ID from filename if not provided
    if patient_id is None:
        patient_id = Path(segmentation_path).stem.replace('.nii', '')
    
    # Label definitions
    labels = {
        1: {'name': 'GTVp (Primary Tumor)', 'color': '#FF4444', 'opacity': 1.0},
        2: {'name': 'GTVn (Nodal)', 'color': '#44FF44', 'opacity': 1.0}
    }
    
    # Check which labels are present
    unique_labels = np.unique(seg_data).astype(int)
    print(f"  Labels found: {unique_labels}")
    
    # Create meshes for each label
    meshes = {}
    for label_val, label_info in labels.items():
        if label_val in unique_labels:
            print(f"  Creating mesh for {label_info['name']}...")
            mesh = create_mesh_from_label(seg_data, label_val, spacing)
            if mesh is not None:
                meshes[label_val] = mesh
                voxel_count = np.sum(seg_data == label_val)
                print(f"    Mesh created: {mesh.n_points} points, {mesh.n_cells} faces ({voxel_count:,} voxels)")
    
    if not meshes:
        print("  ERROR: No meshes could be created!")
        return None
    
    # ============================================================
    # 1. Create Interactive HTML Visualization
    # ============================================================
    print(f"\nCreating interactive 3D HTML visualization...")
    
    html_path = os.path.join(output_dir, f'{patient_id}_3d_interactive.html')
    
    try:
        plotter_html = pv.Plotter(off_screen=True)
        
        for label_val, mesh in meshes.items():
            label_info = labels[label_val]
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
        print(f"  Warning: Could not create HTML (requires trame): {e}")
        print(f"  Skipping HTML export, continuing with PNG views...")
    
    # ============================================================
    # 2. Create Multi-View PNG Images (6 angles like AortaPlot)
    # ============================================================
    print(f"\nCreating multi-view images...")
    
    view_angles = [0, 60, 120, 180, 240, 300]
    view_images = []
    
    # Calculate combined bounds for consistent view
    all_points = np.vstack([mesh.points for mesh in meshes.values()])
    center = all_points.mean(axis=0)
    
    for angle in view_angles:
        plotter = pv.Plotter(off_screen=True, window_size=[600, 800])
        
        for label_val, mesh in meshes.items():
            label_info = labels[label_val]
            plotter.add_mesh(
                mesh, 
                color=label_info['color'], 
                opacity=label_info['opacity'],
                smooth_shading=True
            )
        
        # Add coordinate axes at corner
        # Get mesh bounds
        bounds = plotter.bounds
        x_len = bounds[1] - bounds[0]
        y_len = bounds[3] - bounds[2]
        z_len = bounds[5] - bounds[4]
        
        axis_scale = 0.3  # 30% of size
        axis_origin = np.array([
            bounds[0] - x_len * 0.15,
            bounds[2] - y_len * 0.15,
            bounds[4] - z_len * 0.05
        ])
        
        # Add axis arrows
        axis_colors = ['#0066FF', '#FF9900', '#00CC00']  # X=Blue, Y=Orange, Z=Green
        axis_labels = ['X', 'Y', 'Z']
        axis_lengths = [x_len * axis_scale, y_len * axis_scale, z_len * axis_scale]
        
        for i, (ax_len, ax_color, ax_label) in enumerate(zip(axis_lengths, axis_colors, axis_labels)):
            direction = np.zeros(3)
            direction[i] = 1.0
            
            # Create arrow
            arrow = pv.Arrow(
                start=axis_origin,
                direction=direction,
                scale=ax_len,
                shaft_radius=0.02,
                tip_radius=0.05,
                tip_length=0.2
            )
            plotter.add_mesh(arrow, color=ax_color)
            
            # Add label
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
        plotter.camera.zoom(0.85)
        
        plotter.set_background('#FFFFFF')
        
        # Save individual view
        view_path = os.path.join(output_dir, f'{patient_id}_view_{angle:03d}.png')
        plotter.screenshot(view_path)
        view_images.append(view_path)
        print(f"    Saved view {angle}°: {view_path}")
        plotter.close()
    
    # ============================================================
    # 3. Create Combined Multi-View Image (2x3 grid)
    # ============================================================
    print(f"\nCreating combined multi-view image...")
    
    try:
        from PIL import Image
        
        # Load all view images
        images = [Image.open(p) for p in view_images]
        
        # Create 2x3 grid
        img_width, img_height = images[0].size
        grid_width = img_width * 3
        grid_height = img_height * 2
        
        combined = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
        
        for idx, img in enumerate(images):
            row = idx // 3
            col = idx % 3
            combined.paste(img, (col * img_width, row * img_height))
        
        combined_path = os.path.join(output_dir, f'{patient_id}_multiview.png')
        combined.save(combined_path, dpi=(150, 150))
        print(f"  Saved: {combined_path}")
        
    except ImportError:
        print("  Warning: PIL not available, skipping combined image")
    
    # ============================================================
    # 4. Save VTK meshes for further use
    # ============================================================
    print(f"\nSaving VTK meshes...")
    
    for label_val, mesh in meshes.items():
        label_info = labels[label_val]
        vtk_path = os.path.join(output_dir, f'{patient_id}_label{label_val}_{label_info["name"].split()[0]}.vtk')
        mesh.save(vtk_path)
        print(f"  Saved: {vtk_path}")
    
    # ============================================================
    # 5. Print Summary
    # ============================================================
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nPatient: {patient_id}")
    print(f"Output directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  - {patient_id}_3d_interactive.html  (Interactive 3D viewer)")
    print(f"  - {patient_id}_multiview.png        (6-view combined image)")
    print(f"  - {patient_id}_view_XXX.png         (Individual view images)")
    print(f"  - {patient_id}_labelX_*.vtk         (VTK meshes)")
    print("\nLabel colors:")
    for label_val, label_info in labels.items():
        if label_val in meshes:
            print(f"  Label {label_val}: {label_info['name']} - {label_info['color']}")
    
    return {
        'html': html_path,
        'multiview': combined_path if 'combined_path' in dir() else None,
        'views': view_images,
        'meshes': meshes
    }


def main():
    parser = argparse.ArgumentParser(
        description='3D Visualization for Head & Neck Cancer Segmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_hn_segmentation.py -i output_data/CHUM-001/CHUM-001.nii.gz -o visualizations/CHUM-001
  python visualize_hn_segmentation.py -i prediction.nii.gz -o viz/ --patient-id Patient001
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Path to segmentation NIfTI file')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--patient-id', type=str, default=None,
                        help='Patient ID for labeling (default: from filename)')
    parser.add_argument('--ct', type=str, default=None,
                        help='Optional CT scan for reference')
    
    args = parser.parse_args()
    
    # Run visualization
    visualize_hn_segmentation(
        segmentation_path=args.input,
        output_dir=args.output,
        ct_path=args.ct,
        patient_id=args.patient_id
    )


if __name__ == '__main__':
    main()
