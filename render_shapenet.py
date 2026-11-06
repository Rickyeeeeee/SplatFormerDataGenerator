"""Render ShapeNet OBJ models using the shared full-view COLMAP renderer.

Run inside Blender 3.2, with the same arguments as render_full.py.
"""
import json
import os
import sys

import bpy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render_full as renderer


def load_shapenet_object(object_path):
    # Blender's OBJ importer maps ShapeNet's +Y up axis to Blender's +Z up.
    # Do not reset imported rotations: that would undo this axis conversion.
    bpy.ops.import_scene.obj(filepath=object_path, axis_forward='-Z', axis_up='Y')
    if not any(obj.type == 'MESH' for obj in bpy.context.selected_objects):
        raise ValueError('OBJ contains no mesh objects: ' + object_path)


def normalize_shapenet_scene(rotate):
    bbox_min, bbox_max = renderer.scene_bbox()
    extent = max(bbox_max - bbox_min)
    if extent <= 0:
        raise ValueError('Model has a degenerate bounding box')
    scale = 1.0 / extent
    translation = [float(-scale * (lo + hi) / 2) for lo, hi in zip(bbox_min, bbox_max)]
    result = original_normalize('null')
    metadata_path = os.path.join(renderer.args.output_folder, 'metadata.json')
    with open(metadata_path, encoding='utf-8') as handle:
        metadata = json.load(handle)
    metadata['normalization'] = {
        'obj_axis_forward': '-Z', 'obj_axis_up': 'Y',
        'scale': float(scale), 'translation': translation,
        'coordinate_frame': 'Blender world after OBJ import',
    }
    with open(metadata_path, 'w', encoding='utf-8') as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
    return result


original_normalize = renderer.normalize_scene
renderer.load_object = load_shapenet_object
renderer.get_rotation = lambda _path: 'null'
renderer.normalize_scene = normalize_shapenet_scene

if __name__ == '__main__':
    renderer.main()
