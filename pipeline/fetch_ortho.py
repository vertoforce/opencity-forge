#!/usr/bin/env python3
"""Aerial ortho over any bbox: USDA NAIP through the USGS National Map.

The source is the National Map ImageServer (imagery.nationalmap.gov USGSNAIPImagery),
an official public REST API, no account, no scraping.  NAIP is public domain (USDA FSA).
The newest USDA acquisition over the box is locked (one year, one sensor, no mosaic of
years), exported in the area's UTM CRS at the scene's own GSD with nearest-neighbour
sampling and raw bands (rendering rule None): 4 bands R, G, B, NIR.  This service can
hold an older year for a state than the newest NAIP release published elsewhere.

    fetch_ortho.py --bbox=W,S,E,N --out DIR [--year Y] [--gsd M] [--epsg N] [--pad 40]

Tiles are GeoTIFFs in DIR (any CRS; the readers transform per tile), with the
source, year and licence in each tile's tags and in DIR/source.json.

A local ortho (a city or county aerial) is not fetched here.  Put its GeoTIFF tiles
and a source.json in DIR yourself and declare it in area.yaml sources.ortho; see the
README for the `restricted:` flag its terms may need.
"""
import argparse, json, math, os, sys, time
try:
    import opencity_paths as OCP                # host roots, env-overridable
except ImportError:                            # imported with lib/ off sys.path
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib'))
    import opencity_paths as OCP
if __name__ == '__main__':                       # run as ./fetch_ortho.py under /usr/bin/env python3
    OCP.reexec_in_venv()
import pyproj, requests, rasterio
from rasterio.transform import Affine
from rasterio.crs import CRS

NAIP_SVC = 'https://imagery.nationalmap.gov/arcgis/rest/services/USGSNAIPImagery/ImageServer'


def fetch_naip(a, w, s, e, n):
    """NAIP 4-band over the box (+pad), newest USDA year, native GSD, area UTM CRS."""
    os.makedirs(a.out, exist_ok=True)
    S = requests.Session()
    S.headers['User-Agent'] = OCP.USER_AGENT
    lat0, lon0 = (s + n) / 2, (w + e) / 2
    epsg = a.epsg or ((32600 if lat0 >= 0 else 32700) + int((lon0 + 180) // 6) + 1)
    q = dict(where='Category=1', geometry='%.7f,%.7f,%.7f,%.7f' % (w, s, e, n), geometryType='esriGeometryEnvelope',
             inSR=4326, spatialRel='esriSpatialRelIntersects', returnGeometry='false', f='json',
             outFields='OBJECTID,Name,Year,State,acquisition_date,resolution_value,resolution_units,band_count,agency')
    j = S.get(NAIP_SVC + '/query', params=q, timeout=120).json()
    if 'error' in j:
        raise SystemExit('NAIP catalog query failed: %s' % j['error'])
    rows = [f['attributes'] for f in j.get('features', [])
            if (f['attributes'].get('agency') or '').upper() == 'USDA' and f['attributes'].get('Year')]
    if a.year:
        rows = [r for r in rows if r['Year'] == a.year]
    if not rows:
        raise SystemExit('no USDA NAIP scene over this box in %s' % NAIP_SVC)
    year = max(r['Year'] for r in rows)
    pick = [r for r in rows if r['Year'] == year]
    gsd = a.gsd or min(float(r['resolution_value']) for r in pick)
    print('NAIP %d (%s), %d scene(s), %.2f m, %d bands -> EPSG:%d' % (
        year, pick[0].get('State'), len(pick), gsd, pick[0].get('band_count') or 4, epsg))
    t = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    xs, ys = t.transform([w, e, e, w], [s, s, n, n])
    x0 = math.floor((min(xs) - a.pad) / gsd) * gsd; x1 = math.ceil((max(xs) + a.pad) / gsd) * gsd
    y0 = math.floor((min(ys) - a.pad) / gsd) * gsd; y1 = math.ceil((max(ys) + a.pad) / gsd) * gsd
    Wp, Hp = int(round((x1 - x0) / gsd)), int(round((y1 - y0) / gsd))
    nc, nr = math.ceil(Wp / a.maxpx), math.ceil(Hp / a.maxpx)
    mosaic = json.dumps(dict(mosaicMethod='esriMosaicLockRaster', lockRasterIds=[r['OBJECTID'] for r in pick]))
    out = []
    for ri in range(nr):
        for ci in range(nc):
            c0, c1 = round(ci * Wp / nc), round((ci + 1) * Wp / nc)
            r0, r1 = round(ri * Hp / nr), round((ri + 1) * Hp / nr)
            bx0, bx1 = x0 + c0 * gsd, x0 + c1 * gsd
            by1, by0 = y1 - r0 * gsd, y1 - r1 * gsd
            path = '%s/tile_r%d_c%d.tif' % (a.out, ri, ci)
            if os.path.exists(path):
                print('  have', os.path.basename(path)); out.append(path); continue
            p = dict(bbox='%.3f,%.3f,%.3f,%.3f' % (bx0, by0, bx1, by1), bboxSR=epsg, imageSR=epsg,
                     size='%d,%d' % (c1 - c0, r1 - r0), format='tiff', pixelType='U8', bandIds='0,1,2,3',
                     interpolation='RSP_NearestNeighbor', mosaicRule=mosaic,
                     renderingRule=json.dumps(dict(rasterFunction='None')), f='image')
            for attempt in range(4):
                try:
                    rr = S.get(NAIP_SVC + '/exportImage', params=p, timeout=300)
                    assert rr.ok and rr.headers.get('content-type', '').startswith('image'), (rr.status_code, rr.text[:200])
                    break
                except Exception as ex:
                    print('   retry %d: %s' % (attempt, ex)); time.sleep(4 * (attempt + 1))
            else:
                raise SystemExit('failed NAIP tile r%d c%d' % (ri, ci))
            tmp = path + '.part.tif'; open(tmp, 'wb').write(rr.content)
            with rasterio.open(tmp) as src:
                arr = src.read()
            assert arr.shape[1:] == (r1 - r0, c1 - c0) and arr.shape[0] >= 3, (arr.shape, r1 - r0, c1 - c0)
            with rasterio.open(path, 'w', driver='GTiff', height=r1 - r0, width=c1 - c0, count=arr.shape[0],
                               dtype='uint8', crs=CRS.from_epsg(epsg), transform=Affine(gsd, 0, bx0, 0, -gsd, by1),
                               compress='deflate', predictor=2, tiled=True, blockxsize=512, blockysize=512) as d:
                d.write(arr)
                d.update_tags(SOURCE=NAIP_SVC, YEAR=str(year), GSD_M=str(gsd), BANDS=','.join(['R', 'G', 'B', 'NIR'][:arr.shape[0]]),
                              LICENCE='USDA NAIP, public domain', SCENES=','.join(r['Name'] for r in pick))
            os.remove(tmp)
            print('  r%d c%d %dx%d  %.1f MB  mean RGB %.1f' % (ri, ci, c1 - c0, r1 - r0, len(rr.content) / 1e6, arr[:3].mean()))
            out.append(path)
            time.sleep(1.0)                                  # polite: one export per second
    json.dump(dict(source='naip', service=NAIP_SVC, year=year, gsd_m=gsd, epsg=epsg, bands=['R', 'G', 'B', 'NIR'],
                   licence='USDA NAIP, public domain (attribution appreciated: USDA Farm Production and Conservation)',
                   scenes=[dict(name=r['Name'], year=r['Year'], state=r.get('State'), res=r.get('resolution_value'),
                                acquired=time.strftime('%Y-%m-%d', time.gmtime((r.get('acquisition_date') or 0) / 1000)))
                           for r in pick],
                   bbox_wsen=[w, s, e, n], pad_m=a.pad, tiles=out, fetched=time.strftime('%Y-%m-%dT%H:%M:%S')),
              open(os.path.join(a.out, 'source.json'), 'w'), indent=1)
    print('tiles:', len(out))


ap = argparse.ArgumentParser()
ap.add_argument('--bbox', required=True, help='W,S,E,N (write --bbox=...)')
ap.add_argument('--out', required=True)
ap.add_argument('--source', default='naip', choices=['naip'])
ap.add_argument('--year', type=int, help='lock this year instead of the newest')
ap.add_argument('--gsd', type=float, help='output GSD (default: the scene\'s own)')
ap.add_argument('--epsg', type=int, help='output CRS (default: the box\'s UTM zone)')
ap.add_argument('--pad', type=float, default=40.0, help='metres around the box')
ap.add_argument('--maxpx', type=int, default=4000)
a = ap.parse_args()
w, s, e, n = (float(v) for v in a.bbox.split(','))
fetch_naip(a, w, s, e, n)
