# How opencity-forge works

The pipeline turns four open datasets into one textured glTF model of a small box.

- Lidar gives the ground and the heights.
- OpenStreetMap gives the footprints, paths, water and land use.
- Street-level 360 panoramas give the colour of every wall and most of the ground.
- An aerial photo fills the ground the panoramas never saw.

Most steps are stage scripts, `stages/sNN_*.py`.

## The stages, in order

`opencity build` runs these steps top to bottom; `POSES_STEPS` and `POSES_STAGES` in `pipeline/area_cli.py` give the exact list.

| Step | Stages | What it does |
|---|---|---|
| Fetch | `fetch/fetch_cc0.py`, `pipeline/fetch_panos.py`, `fetch/fetch_context.py`, `pipeline/fetch_ortho.py` | the shared CC0 material tiles (once), Mapillary panoramas and metadata, a USGS 3DEP lidar crop, OSM features and a NAIP ortho for the box (`opencity fetch`, before the build) |
| Frame | `pipeline/nomesh.py manifest`, `fetch/fuse_context.py` | one local ENU frame; lidar heights about the origin |
| Poses | `pipeline/nomesh.py poses`, `empty-geom`, `dsm` | each panorama at its Mapillary position, height set from the lidar ground; a zero-face photo-mesh stand-in; lidar surface grids (DSM and DTM) |
| Knobs | `pipeline/derive_knobs.py` | measures water level, path widths, lidar spacing and similar values from the area's own data into `derived.yaml` |
| Context | `s02_osm`, `s03b_city_trees`, `s03_trees`, `s04_tree_merge`, `s05_inventory`, `s05b_inventory_join` | OSM layers by tag rule; trees from lidar crowns, OSM and city inventories |
| Geometry | `s10a_3dfier` (with `s10a_layers`) | 3dfier (in Docker) welds footprints, ground, water and decks into one surface; layers are split and repaired |
| Colour | `s19_delight`, `s40_masks`, `s20_project` | per-panorama exposure and white balance; transient masks; panoramas projected onto every face |
| Ground | `pipeline/nomesh_ground.py tint`, `s21_material`, `pipeline/nomesh_ground.py classes`, `s50_classtint` | ground colour from the panoramas, with the ortho for cells no panorama saw; a material class per face; per-class colour |
| Facades | `s33_walls`, `s34_wallproj`, `s35_facadetex`, `s60_wall_bestview`, then `s35` and the `s60` bake, relevel and atlas passes again, then `s62_wall_integrate` (via `pipeline/nomesh_facades.py swap`) | wall rectangles, per-texel projection, one baked texture per wall |
| Objects | `pipeline/city_props.py`, `s80_furniture_inventory`, `s81_prop_assets`, `lib/objects/objassets.py`, `s84_object_inventory` | street furniture, props and vehicles found and their CC0 kits baked |
| Composite | `s22_composite` | one tiling CC0 material per class, tinted by the measured colour |
| Swaps | `pipeline/nomesh_swaps.py`: `s82_prop_swap`, `s85_object_place`, `s94_ground_joints` | props and objects placed as sized nodes or CC0 models; sidewalk joints |
| Web copy | `pipeline/area_cli.py` | `model_web.glb` (`gltf-transform resize`) |

## Rules the stages follow

- **One photograph per patch.** Even after the per-patch alignment (`s60_wall_bestview refine`), the panoramas that see a wall disagree by 10 to 25 cm. A median of several views is a box blur, so each patch takes one owner view.
- **No stage reads a file a later stage writes.**
- **A cache never keys on a wall id.** Wall ids move when geometry changes.
- **Local data is optional.** Every input has a national fallback (NAIP, USGS 3DEP, OSM). A local source writes an empty output outside its envelope.
- **Knobs are measured, not copied.** An area's knobs come from its own data. A hand override goes in `knobs:` in `area.yaml` with a `reason:`.
- **Fail loudly.** A missing derived value stops the stage that needs it instead of falling back to a hard-coded default. The one stated exception is the tree allometry prior in `s05b`, used when an area has under 8 matched trees.

<details><summary>Colour: what a stage writes vs what you see</summary>

- Colours are linear RGB inside the pipeline. sRGB appears only in renders and textures.
- `COLOR_0` (the vertex tint) multiplies the material's base colour. A tint above 1.0 is clipped by the viewer, so lifts are capped.
- Dark tints have a floor. Without it, shadowed panorama pixels would paint lawns and walls near black.
- Water uses its own measured colour ramp. Its tint and its texture come from the same numbers.
</details>

## Gates

`./opencity check <area>` runs the gates on a built model.

- `model.glb` and `model_web.glb` exist, and the model covers the area's box.
- It has geometry.
- `gltf-transform validate` reports no errors.
- The fuse_context selfcheck passes, when the manifest records one.
- `derived.yaml` exists.

<details><summary>Area config</summary>

- `areas/<name>/area.yaml` holds the box, CRS, sources, paths and knobs. `pipeline/area_config.schema.json` describes it.
- `OPENCITY_AREA=<name>` makes `lib/common.py` load the config. `opencity build` sets it for every stage.
- Paths in the config are `${ROOT}` templates (`${WORK}`, `${DATA}`), resolved by `lib/opencity_paths.py`.
- `license: public` restricts the build to public sources.
</details>
