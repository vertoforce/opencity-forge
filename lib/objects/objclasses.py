"""The class table for the s84 object inventory.

One place that says, for every object kind the three sensors can see in an
area: what the Mapillary vocabulary calls it, how big it is when nothing
measures it, whether it is a fixed part of the site (static) or present only at
capture time (transient), and which CC0 asset -- if any -- s85 can place for it.

`box_m` is [length, width, height] in metres and is the PLACEHOLDER box a
consumer substitutes a model into when `asset` is null.  Sizes are the ordinary
real-world size of the thing, not a measurement of any area's instances; every
row in the inventory that got measured carries its own footprint instead.

Kinds already covered by the s80 furniture inventory (bench, lamp_post,
bollard, trash_can, bike_rack, picnic_table, sign_post, flagpole,
railing_segment, mooring_post, small_boat, fire_hydrant, mast, drinking_water)
are deliberately NOT re-detected here; s84 loads prop_inventory.json and treats
those rows as already-placed, so an object is only new if nothing in there
explains it.
"""

# Mapillary detection value -> (kind, subkind).  People, riders and animals are
# excluded by construction; so are markings, flat surfaces and the void-* and
# nature-* classes, which are surfaces rather than objects.
PHOTO_MAP = {
    'object--vehicle--car':            ('car', 'car'),
    'object--vehicle--truck':          ('truck', 'truck'),
    'object--vehicle--bus':            ('bus', 'bus'),
    'object--vehicle--caravan':        ('truck', 'caravan'),
    'object--vehicle--trailer':        ('truck', 'trailer'),
    'object--vehicle--other-vehicle':  ('car', 'other_vehicle'),
    'object--vehicle--motorcycle':     ('motorcycle', 'motorcycle'),
    'object--vehicle--bicycle':        ('bicycle', 'bicycle'),
    'object--vehicle--wheeled-slow':   ('bicycle', 'wheeled_slow'),
    'object--vehicle--boat':           ('boat', 'boat'),
    'object--vehicle--on-rails':       ('rail_vehicle', 'streetcar'),
    'object--support--utility-pole':   ('utility_pole', 'utility_pole'),
    'object--support--traffic-sign-frame': ('sign_frame', 'sign_frame'),
    'object--traffic-light--general-upright-front': ('traffic_light', 'vehicle'),
    'object--traffic-light--general-upright-back':  ('traffic_light', 'vehicle'),
    'object--traffic-light--general-upright-side':  ('traffic_light', 'vehicle'),
    'object--traffic-light--general-upright':       ('traffic_light', 'vehicle'),
    'object--traffic-light--general-horizontal-front': ('traffic_light', 'vehicle'),
    'object--traffic-light--pedestrians-front':     ('traffic_light', 'pedestrian'),
    'object--traffic-light--pedestrians-side':      ('traffic_light', 'pedestrian'),
    'object--traffic-light--pedestrians-back':      ('traffic_light', 'pedestrian'),
    'object--traffic-light--pedestrians':           ('traffic_light', 'pedestrian'),
    'object--sign--advertisement':     ('sign_board', 'advertisement'),
    'object--sign--store':             ('sign_board', 'store'),
    'object--sign--information':       ('sign_board', 'information'),
    'object--sign--other':             ('sign_board', 'other'),
    'object--banner':                  ('banner', 'banner'),
    'object--junction-box':            ('utility_cabinet', 'junction_box'),
    'object--cctv-camera':             ('cctv_camera', 'cctv_camera'),
    'object--parking-meter':           ('parking_meter', 'parking_meter'),
    'object--mailbox':                 ('mailbox', 'mailbox'),
    'object--phone-booth':             ('phone_booth', 'phone_booth'),
    'object--traffic-cone':            ('traffic_cone', 'traffic_cone'),
    'object--manhole':                 ('ground_fixture', 'manhole'),
    'object--catch-basin':             ('ground_fixture', 'catch_basin'),
    'object--water-valve':             ('ground_fixture', 'water_valve'),
    'construction--barrier--concrete-block': ('jersey_barrier', 'concrete_block'),
}
# every regulatory--*, warning--*, information--*, complementary--* value is a
# traffic sign face; the subkind is the sign's own name, which is how the
# "signs by type" count is produced.
SIGN_PREFIXES = ('regulatory--', 'warning--', 'information--', 'complementary--')
SIGN_KIND = 'traffic_sign_face'

# which prop class each new kind may be confirming
CONFIRMS = {'utility_pole': ('lamp_post', 'sign_post', 'mast', 'flagpole'),
            'sign_frame': ('sign_post',),
            'traffic_sign_face': ('sign_post',),
            'boat': ('small_boat',),
            'sign_board': ('sign_post',)}

# kind -> spec.
#   box_m       placeholder [length, width, height]
#   static      True = fixed part of the site, False = transient (present only at capture time)
#   dz          allowed height of a triangulated point above the model floor
#   min_views   panoramas needed when photos are the only source
#   match_m     fusion radius
#   asset       key in the s85 asset library, or None (placeholder only)
#   water_ok    may stand on water
SPEC = {
    'car':            dict(box_m=[4.5, 1.85, 1.5], static=False, dz=(-1.0, 2.0),
                           min_views=6, match_m=3.0, asset='car', water_ok=False),
    'truck':          dict(box_m=[7.0, 2.4, 3.0], static=False, dz=(-1.0, 3.5),
                           min_views=6, match_m=4.0, asset='truck', water_ok=False),
    'bus':            dict(box_m=[12.0, 2.6, 3.2], static=False, dz=(-1.0, 3.5),
                           min_views=6, match_m=5.0, asset='bus', water_ok=False),
    'motorcycle':     dict(box_m=[2.1, 0.8, 1.2], static=False, dz=(-1.0, 1.5),
                           min_views=8, match_m=2.0, asset=None, water_ok=False),
    'bicycle':        dict(box_m=[1.8, 0.6, 1.1], static=False, dz=(-1.0, 1.5),
                           min_views=8, match_m=2.0, asset=None, water_ok=False),
    'boat':           dict(box_m=[8.0, 2.8, 2.5], static=False, dz=(-2.5, 3.0),
                           min_views=6, match_m=6.0, asset='boat', water_ok=True),
    'rail_vehicle':   dict(box_m=[20.0, 2.6, 3.6], static=False, dz=(-1.0, 4.0),
                           min_views=10, match_m=8.0, asset=None, water_ok=False),
    'utility_pole':   dict(box_m=[0.4, 0.4, 9.0], static=True, dz=(-1.0, 10.0),
                           min_views=12, match_m=6.0, asset='utility_pole',
                           water_ok=False),
    # one signal POLE, not one signal head: the vocabulary fires per head and an
    # intersection carries four to eight of them within a couple of metres.
    'traffic_light':  dict(box_m=[0.4, 0.4, 4.0], static=True, dz=(-1.0, 7.0),
                           min_views=12, match_m=12.0, asset='traffic_light',
                           water_ok=False),
    'sign_frame':     dict(box_m=[2.0, 0.3, 4.0], static=True, dz=(-1.0, 6.0),
                           min_views=8, match_m=3.0, asset=None, water_ok=False),
    'traffic_sign_face': dict(box_m=[0.8, 0.1, 0.8], static=True, dz=(-1.0, 5.0),
                           min_views=8, match_m=2.5, asset=None, water_ok=False),
    'sign_board':     dict(box_m=[1.6, 0.2, 2.2], static=True, dz=(-1.0, 6.0),
                           min_views=10, match_m=4.0, asset=None, water_ok=False),
    'banner':         dict(box_m=[1.0, 0.1, 2.0], static=True, dz=(0.0, 7.0),
                           min_views=8, match_m=3.0, asset=None, water_ok=False),
    'utility_cabinet': dict(box_m=[0.9, 0.5, 1.2], static=True, dz=(-1.0, 2.0),
                           min_views=6, match_m=2.5, asset='utility_cabinet',
                           water_ok=False),
    'cctv_camera':    dict(box_m=[0.3, 0.3, 0.3], static=True, dz=(0.0, 9.0),
                           min_views=6, match_m=3.0, asset=None, water_ok=False),
    'parking_meter':  dict(box_m=[0.3, 0.3, 1.4], static=True, dz=(-1.0, 2.0),
                           min_views=6, match_m=2.0, asset=None, water_ok=False),
    'mailbox':        dict(box_m=[0.6, 0.5, 1.3], static=True, dz=(-1.0, 2.0),
                           min_views=5, match_m=2.0, asset=None, water_ok=False),
    'phone_booth':    dict(box_m=[1.0, 1.0, 2.2], static=True, dz=(-1.0, 3.0),
                           min_views=4, match_m=2.5, asset=None, water_ok=False),
    'traffic_cone':   dict(box_m=[0.4, 0.4, 0.7], static=False, dz=(-1.0, 1.0),
                           min_views=6, match_m=1.5, asset=None, water_ok=False),
    'ground_fixture': dict(box_m=[0.7, 0.7, 0.05], static=True, dz=(-1.0, 0.6),
                           min_views=5, match_m=1.5, asset=None, water_ok=False),
    'jersey_barrier': dict(box_m=[2.0, 0.6, 0.9], static=True, dz=(-1.0, 1.5),
                           min_views=8, match_m=2.5, asset=None, water_ok=False),
    # kinds no photo vocabulary names: found by OSM tag, lidar cluster or the
    # aerial blob pass only.
    'planter':        dict(box_m=[1.2, 1.2, 0.9], static=True, dz=(-1.0, 1.5),
                           min_views=99, match_m=2.0, asset='planter', water_ok=False),
    'kiosk':          dict(box_m=[3.0, 2.5, 2.8], static=True, dz=(-1.0, 3.5),
                           min_views=99, match_m=4.0, asset=None, water_ok=False),
    'shelter':        dict(box_m=[4.0, 3.0, 2.8], static=True, dz=(-1.0, 4.0),
                           min_views=99, match_m=5.0, asset=None, water_ok=False),
    'artwork':        dict(box_m=[1.5, 1.5, 2.5], static=True, dz=(-1.0, 6.0),
                           min_views=99, match_m=4.0, asset=None, water_ok=False),
    'playground_item': dict(box_m=[3.0, 3.0, 2.5], static=True, dz=(-1.0, 4.0),
                           min_views=99, match_m=4.0, asset=None, water_ok=False),
    'unknown_cluster': dict(box_m=[1.0, 1.0, 1.2], static=True, dz=(-1.0, 4.0),
                           min_views=99, match_m=2.0, asset=None, water_ok=False),
}

# OSM tags that name an object this stage owns (the s80 table owns the rest).
OSM_RULES = [
    (('leisure', 'playground'), 'playground_item', 'playground'),
    (('playground', None), 'playground_item', None),
    (('attraction', None), 'playground_item', None),
    (('amenity', 'shelter'), 'shelter', 'shelter'),
    (('tourism', 'artwork'), 'artwork', 'artwork'),
    (('historic', 'memorial'), 'artwork', 'memorial'),
    (('man_made', 'bell'), 'artwork', 'bell'),
    (('man_made', 'surveillance'), 'cctv_camera', 'surveillance'),
    (('amenity', 'vending_machine'), 'kiosk', 'vending_machine'),
    (('amenity', 'kiosk'), 'kiosk', 'kiosk'),
    (('shop', 'kiosk'), 'kiosk', 'kiosk'),
    (('amenity', 'parking'), None, None),          # area, used as a car prior
]


def photo_kind(value):
    """-> (kind, subkind) or None for a Mapillary detection value."""
    if value in PHOTO_MAP:
        return PHOTO_MAP[value]
    for p in SIGN_PREFIXES:
        if value.startswith(p):
            return (SIGN_KIND, value.split('--g')[0][len(p):] or value)
    return None
