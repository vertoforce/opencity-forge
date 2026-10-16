"""An optional city street-tree inventory for the model bbox -> inventory/city_trees.json.

A published point inventory places trees better than lidar crown segmentation alone:
the point gives the position, species and trunk diameter, and lidar gives the height
(s05b joins the two).  pipeline/city_trees.py lists the cities whose open tree layers
are supported, each as an envelope, a bbox-only fetch and a row mapper.

Outside every listed city, or when area.yaml sets sources.trees.city: none, this stage
writes an empty inventory so s05b keeps its input, and the lidar crowns (s03/s04/s05)
plus OSM carry every tree.  A city source that fails (network, schema) falls back the
same way, loudly, instead of failing the build.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
import common as C

INV = C.W + '/inventory'
# model extent in ENU metres (x0, x1, y0, y1): the area's own box
EXTENT = (0.0, 0.0, 0.0, 0.0)
if hasattr(C, '_areacfg') and C._areacfg.build_area_cfg(C) is not None:
    _e = C._areacfg.enu_ext(C); EXTENT = (_e[0], _e[2], _e[1], _e[3])


def empty_doc():
    return dict(what="no city tree inventory for this area; lidar crowns and OSM only",
                fetched=time.strftime('%Y-%m-%dT%H:%M:%S'), bbox_wsen=list(C.BBOX_WSEN),
                extent_enu_m=list(EXTENT), frame=dict(type="ENU metres", origin_lat_lon_alt=list(C.ORIGIN)),
                endpoints={}, licence=None,
                counts=dict(combined_in_bbox=0, combined_in_extent=0,
                            with_species=0, with_dbh=0, with_field_height=0, by_dept={}),
                trees=[])


def main():
    os.makedirs(INV, exist_ok=True)
    cfg = C._areacfg.build_area_cfg(C) if hasattr(C, '_areacfg') else None
    if cfg is None:
        sys.exit('s03b: OPENCITY_AREA must name an area whose config sets paths.scratch_dir')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'pipeline'))
    import city_trees
    try:
        doc = city_trees.build(C, EXTENT, cfg)
    except Exception as e:     # a local source is optional: fall back loudly, never fail the build
        print(f's03b: city tree inventory FAILED ({str(e)[:200]}); falling back to lidar crowns + OSM', flush=True)
        C.note('tree_inventory_city_error', str(e)[:300]); doc = None
    if doc is not None:
        json.dump(doc, open(INV + '/city_trees.json', 'w'), indent=1)
        print(f"s03b: {doc['city']} inventory, {doc['counts']['combined_in_extent']} trees in the extent")
        C.note('tree_inventory_source', doc['counts'] | dict(city=doc['city'], licence=doc['licence']))
    else:
        doc = empty_doc()
        json.dump(doc, open(INV + '/city_trees.json', 'w'), indent=1)
        print('s03b: no city tree inventory for this area; wrote an empty one')
        C.note('tree_inventory_source', doc['counts'])
    C.mark('city_trees')


if __name__ == '__main__':
    main()
