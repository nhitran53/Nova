"""Build one HiRISE test patch in the existing CTX coordinate system."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from convert_jezero import ROOT, OUTPUT, geo_keys, sample


DTM = "JEZ_hirise_soc_006_DTM_MOLAtopography_DeltaGeoid_1m_Eqc_latTs0_lon0_blend40.tif"
ORTHO = "JEZ_hirise_soc_006_orthoMosaic_25cm_Eqc_latTs0_lon0_first.tif"
DESTINATION = OUTPUT / "HiRISE"
RESOLUTION = 513
CELLS = 32
COLLAR = 64.0
GROUND = np.array([0.48, 0.34, 0.24])


class Raster:
    def __init__(self, path):
        self.path = path
        with Image.open(path) as image:
            self.tags = dict(image.tag_v2)
            self.width, self.height = image.size
            mode = image.mode
        t = self.tags
        k = geo_keys(t)
        if (k.get(1024) != 1 or k.get(1025) != 1 or k.get(3075) != 17 or
                k.get(3076) != 9001 or k.get(2054) != 9102 or 34264 in t):
            raise ValueError(f"{path}: expected north-up PixelIsArea equirectangular meters/degrees")
        if mode not in ("F", "L") or t.get(277) != 1 or t.get(259) != 1 or 273 not in t:
            raise ValueError(f"{path}: expected an uncompressed single-band float32/uint8 strip TIFF")
        offsets, counts = t[273], t[279]
        dtype = np.dtype("<f4" if mode == "F" else "u1")
        with path.open("rb") as stream:
            if stream.read(2) != b"II":
                raise ValueError("This reader requires little-endian TIFFs")
        if (sum(counts) != self.width * self.height * dtype.itemsize or
                not all(a + b == c for a, b, c in zip(offsets, counts, offsets[1:]))):
            raise ValueError("Raster strips must be contiguous; use GDAL for other TIFF layouts")
        self.data = np.memmap(path, mode="r", dtype=dtype, offset=offsets[0], shape=(self.height, self.width))
        self.nodata = float(t[42113])
        self.sx, self.sy, _ = t[33550]
        px, py, _, east, north, _ = t[33922]
        self.west, self.north = east - px * self.sx, north + py * self.sy
        if self.sx <= 0 or self.sy <= 0:
            raise ValueError("Pixel spacing must be positive")
        self.projection = dict(sphereRadius=k[2057], standardParallel=k[3078],
                               centralMeridian=k[3088], latitudeOfOrigin=k[3089],
                               falseEasting=k[3082], falseNorthing=k[3083])
        if k[2057] != k[2058] or k.get(2061) != 0:
            raise ValueError("Expected a Mars sphere with the zero reference meridian")

    def inspect(self):
        minimum, maximum, missing = float("inf"), -float("inf"), 0
        for row in range(0, self.height, 256):
            a = self.data[row:row + 256]
            valid = np.isfinite(a) & (a != self.nodata)
            missing += int((~valid).sum())
            if valid.any():
                minimum = min(minimum, float(a[valid].min()))
                maximum = max(maximum, float(a[valid].max()))
        source_path = self.path.relative_to(ROOT).as_posix() if self.path.is_relative_to(ROOT) else str(self.path)
        return dict(path=source_path, width=self.width, height=self.height,
                    pixelSizeX=self.sx, pixelSizeY=self.sy, west=self.west, north=self.north,
                    east=self.west + self.width * self.sx, south=self.north - self.height * self.sy,
                    noData=self.nodata, noDataPixels=missing, minimum=minimum, maximum=maximum,
                    bytes=self.path.stat().st_size, **self.projection)

    def read(self, east, north):
        xs = (east - self.west) / self.sx - .5
        ys = (self.north - north) / self.sy - .5
        if min(xs) < 0 or max(xs) > self.width - 1 or min(ys) < 0 or max(ys) > self.height - 1:
            raise ValueError("The selected patch extends outside the source raster")
        x0, x1 = int(np.floor(min(xs))), int(np.ceil(max(xs))) + 1
        y0, y1 = int(np.floor(min(ys))), int(np.ceil(max(ys))) + 1
        window = np.array(self.data[y0:y1, x0:x1])
        valid = np.isfinite(window) & (window != self.nodata)
        heights, present = sample(window, valid, xs - x0, ys - y0, 0)
        if not present.all():
            raise ValueError("The test patch contains missing source data; select another center")
        return heights


def to_source(east, north, ctx, source):
    a, b = ctx, source.projection
    if a["sphereRadius"] != b["sphereRadius"]:
        raise ValueError("Mars reference radii differ; an explicit datum transformation is required")
    radius = a["sphereRadius"]
    longitude = ((east - a["falseEasting"]) / (radius * math.cos(math.radians(a["standardParallel"]))) +
                 math.radians(a["centralMeridian"]))
    latitude = (north - a["falseNorthing"]) / radius + math.radians(a["latitudeOfOrigin"])
    return (b["falseEasting"] + radius * (longitude - math.radians(b["centralMeridian"])) *
            math.cos(math.radians(b["standardParallel"])),
            b["falseNorthing"] + radius * (latitude - math.radians(b["latitudeOfOrigin"])))


def blend_weight(xs, zs, width, length):
    edge = np.minimum(np.minimum(xs, width - xs)[None, :], np.minimum(zs, length - zs)[:, None])
    t = np.clip(edge / COLLAR, 0, 1)
    return t * t * (3 - 2 * t)


def find_source(name):
    for folder in (ROOT / "Data/Source", ROOT):
        path = folder / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Place {name} in Data/Source, or supply its path")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dtm", type=Path)
    parser.add_argument("--ortho", type=Path)
    parser.add_argument("--center-east", type=float, default=4354493.764,
                        help="Center easting in the CTX projection; default is the existing player location")
    parser.add_argument("--center-north", type=float, default=1093299.863)
    args = parser.parse_args()
    Image.MAX_IMAGE_PIXELS = None
    manifest = OUTPUT / "terrain.json"
    ctx = json.loads(manifest.read_text())
    if ctx["resolution"] != 1025 or ctx["columns"] != 5 or ctx["rows"] != 5:
        raise ValueError("Regenerate the existing 25-tile CTX terrain first")
    dtm = Raster((args.dtm or find_source(DTM)).resolve())
    ortho = Raster((args.ortho or find_source(ORTHO)).resolve())
    if dtm.data.dtype != np.dtype("<f4") or ortho.data.dtype != np.dtype("u1"):
        raise ValueError("Expected float elevations and an 8-bit grayscale orthomosaic")
    sources = [r.inspect() for r in (dtm, ortho)]
    for info in sources:
        print(json.dumps(info), flush=True)
    spacing_x, spacing_z = ctx["sampleSpacingX"], ctx["sampleSpacingZ"]
    start_x = round((args.center_east - ctx["west"]) / spacing_x) - CELLS // 2
    start_z = round((args.center_north - ctx["south"]) / spacing_z) - CELLS // 2
    tile_x, local_x = divmod(start_x, 1024)
    tile_z, local_z = divmod(start_z, 1024)
    if not (0 <= tile_x < 5 and 0 <= tile_z < 5 and 0 < local_x < 1024 - CELLS and 0 < local_z < 1024 - CELLS):
        raise ValueError("This small test area must fit strictly inside one CTX tile")
    width, length = CELLS * spacing_x, CELLS * spacing_z
    x, z = start_x * spacing_x, start_z * spacing_z
    xs, zs = np.linspace(0, width, RESOLUTION), np.linspace(0, length, RESOLUTION)
    east, north = ctx["west"] + x + xs, ctx["south"] + z + zs
    high = dtm.read(*to_source(east, north, ctx, dtm))
    tile = next(t for t in ctx["tiles"] if (t["x"], t["z"]) == (tile_x, tile_z))
    raw_path = OUTPUT / tile["raw"]
    raw = np.fromfile(raw_path, dtype="<u2").reshape(1025, 1025)
    valid = np.ones(raw.shape, dtype=bool)
    coarse, _ = sample(raw, valid, local_x + xs / spacing_x, local_z + zs / spacing_z, 0)
    coarse = ctx["baseElevation"] + coarse / 65535 * ctx["heightRange"]
    if tile["mask"]:
        surface = np.fromfile(OUTPUT / tile["mask"], dtype="u1").reshape(1024, 1024)
        if not surface[local_z - 1:local_z + CELLS + 1, local_x - 1:local_x + CELLS + 1].all():
            raise ValueError("CTX coverage is missing around this patch")
    weight = blend_weight(xs, zs, width, length)
    heights = coarse + weight * (high - coarse)
    base = math.floor(float(heights.min())) - 1
    height_range = math.ceil(float(heights.max())) + 1 - base
    encoded = np.rint((heights - base) / height_range * 65535).astype("<u2")
    restored = base + encoded.astype(float) / 65535 * height_range
    error = float(np.max(abs(restored - heights)))
    assert error <= height_range / 65535 / 2 + 1e-9
    assert np.max(abs((heights - coarse)[weight == 0])) == 0
    texture_size = 2048
    tx = (np.arange(texture_size) + .5) / texture_size * width
    tz = (np.arange(texture_size) + .5) / texture_size * length
    grey = ortho.read(*to_source(ctx["west"] + x + tx, ctx["south"] + z + tz, ctx, ortho)) / 255
    texture_weight = blend_weight(tx, tz, width, length)
    rgb = GROUND + texture_weight[..., None] * (grey[..., None] - GROUND)
    # PNG rows run north to south; terrain height rows run south to north.
    pixels = np.rint(np.clip(rgb[::-1], 0, 1) * 255).astype("u1")
    DESTINATION.mkdir(parents=True, exist_ok=True)
    encoded.tofile(DESTINATION / "heights.raw")
    Image.fromarray(pixels).save(DESTINATION / "ortho.png")
    delta = high - coarse
    meta = dict(version=1, resolution=RESOLUTION, x=x, z=z, widthMeters=width, lengthMeters=length,
                baseElevation=base, heightRange=height_range, collarMeters=COLLAR,
                ctxTileX=tile_x, ctxTileZ=tile_z, ctxCellX=local_x, ctxCellZ=local_z, ctxCells=CELLS,
                ctxManifestSha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                ctxRawSha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
                ctxRaw=tile["raw"], ctxBaseElevation=ctx["baseElevation"], ctxHeightRange=ctx["heightRange"],
                west=float(east[0]), south=float(north[0]), east=float(east[-1]), north=float(north[-1]),
                sourceInfo=sources, heightMinimum=float(high.min()), heightMaximum=float(high.max()),
                deltaMinimum=float(delta.min()), deltaMedian=float(np.median(delta)), deltaMaximum=float(delta.max()),
                deltaP95Absolute=float(np.percentile(abs(delta), 95)), verticalOffsetApplied=0,
                maxQuantizationError=error, imagery="Grayscale source DN / 255; no invented color or contrast stretch",
                transition="64 m smoothstep collar blends to original quantized CTX; the inner area retains HiRISE heights",
                raw="heights.raw", texture="ortho.png",
                metadataUrls=["https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_dtm_mosaic",
                              "https://astrogeology.usgs.gov/search/map/mars_2020_terrain_relative_navigation_hirise_orthorectified_image_mosaic",
                              "https://astrogeology.usgs.gov/search/map/mars_2020_science_investigation_ctx_dem_mosaic"])
    (DESTINATION / "terrain.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Patch: {width:g} x {length:g} m at Unity X/Z {x:g}, {z:g}; {RESOLUTION} samples", flush=True)
    print(f"HiRISE minus CTX: median {meta['deltaMedian']:.3f} m, range {delta.min():.3f} to {delta.max():.3f} m; no shift applied")
    print(f"Height quantization <= {error:.6f} m; 2048-square image; all source samples valid")


if __name__ == "__main__":
    main()
