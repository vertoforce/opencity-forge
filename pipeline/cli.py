#!/usr/bin/env python3
"""opencity: open map data to a photo-textured 3D model of a small bounding box of a city.

    opencity new-area NAME --bbox=W,S,E,N   write areas/NAME/area.yaml
    opencity fetch AREA [--detach]          Mapillary panoramas, lidar, OSM, aerial photo
    opencity build AREA [--detach]          geometry, textures, the glb
    opencity check AREA                     gates on the built model
    opencity publish AREA                   copy the model to OPENCITY_PUBLISH
    opencity derive AREA                    data-derived knobs only
    opencity stage AREA SCRIPT [ARGS]       run one stage script for an area
    opencity areas                          list configured areas

The commands live in pipeline/area_cli.py; roots in lib/opencity_paths.py.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import opencity_paths as OCP                    # noqa: E402  host roots, env-overridable

if __name__ == '__main__':                     # run as ./opencity under /usr/bin/env python3
    OCP.reexec_in_venv()

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import area_cli                                # noqa: E402

if __name__ == '__main__':
    area_cli.main(sys.argv[1:])
