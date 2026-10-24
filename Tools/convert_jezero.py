"""Convert the Jezero GeoTIFF with Python, NumPy and Pillow."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Data/Source/M20_JezeroCrater_CTXDEM_20m.tif"
OUTPUT = ROOT / "Assets/JezeroTerrain/Generated"
RESOLUTION = 1025


def geo_keys(tags):
    directory = tags[34735]
    keys = {}
    for i in range(4, len(directory), 4):
        key, location, count, offset = directory[i:i + 4]
        value = offset if location == 0 else tags[location][offset:offset + count]
        keys[key] = value[0] if isinstance(value, tuple) and count == 1 else value
    return keys


def sample(elevation, valid, xs, ys, minimum):
    xs = np.clip(xs, 0, elevation.shape[1] - 1)
    ys = np.clip(ys, 0, elevation.shape[0] - 1)
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    x1 = np.minimum(x0 + 1, elevation.shape[1] - 1)
    y1 = np.minimum(y0 + 1, elevation.shape[0] - 1)
    wx, wy = xs - x0, ys - y0
    heights = np.zeros((len(ys), len(xs)), dtype=np.float64)
    usable = np.ones(heights.shape, dtype=bool)
    for xx, yy, weight in (
        (x0, y0, (1 - wy)[:, None] * (1 - wx)),
        (x1, y0, (1 - wy)[:, None] * wx),
        (x0, y1, wy[:, None] * (1 - wx)),
        (x1, y1, wy[:, None] * wx),
    ):
        values = elevation[np.ix_(yy, xx)]
        present = valid[np.ix_(yy, xx)]
        usable &= present | (weight == 0)
        heights += np.where(present, values, minimum) * weight
    heights[~usable] = minimum
    return heights, usable


def main():
    with Image.open(SOURCE) as image:
        tags = dict(image.tag_v2)
        elevation = np.array(image, dtype=np.float32)
    keys = geo_keys(tags)
    if elevation.ndim != 2 or keys.get(1024) != 1 or keys.get(1025) != 1:
        raise ValueError("Expected a single-band, projected, PixelIsArea DEM.")
    if keys.get(3075) != 17 or keys.get(3076) != 9001 or 34264 in tags:
        raise ValueError("Expected the north-up Jezero equirectangular projection in meters.")
    sx, sy, _ = tags[33550]
    px, py, _, east, north, _ = tags[33922]
    west, north = east - px * sx, north + py * sy
    if sx <= 0 or sy <= 0:
        raise ValueError("Pixel spacing must be positive.")
    nodata = float(tags[42113])
    valid = np.isfinite(elevation) & (elevation != nodata)
    minimum, maximum = float(elevation[valid].min()), float(elevation[valid].max())
    height_range = maximum - minimum
    if height_range <= 0:
        raise ValueError("The DEM has no elevation range.")
    height, width = elevation.shape
    width_m, length_m = width * sx, height * sy
    columns = int(np.ceil(width / (RESOLUTION - 1)))
    rows = int(np.ceil(height / (RESOLUTION - 1)))
    # One global grid gives every pair of neighboring tiles identical edge samples.
    xs = np.linspace(0, width, columns * (RESOLUTION - 1) + 1) - 0.5
    ys = height - 0.5 - np.linspace(0, height, rows * (RESOLUTION - 1) + 1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tiles = []
    max_error = 0.0
    output_min, output_max = maximum, minimum
    solid_cells = 0
    for z in range(rows):
        for x in range(columns):
            start_x, start_z = x * (RESOLUTION - 1), z * (RESOLUTION - 1)
            heights, present = sample(elevation, valid,
                                      xs[start_x:start_x + RESOLUTION],
                                      ys[start_z:start_z + RESOLUTION], minimum)
            raw = np.rint((heights - minimum) / height_range * 65535).astype("<u2")
            surface = present[:-1, :-1] & present[1:, :-1] & present[:-1, 1:] & present[1:, 1:]
            name = f"tile_{x}_{z}"
            raw.tofile(OUTPUT / f"{name}.raw")
            mask_name = "" if surface.all() else f"{name}.holes.bytes"
            if mask_name:
                surface.astype(np.uint8).tofile(OUTPUT / mask_name)
            restored = minimum + raw[present].astype(np.float64) / 65535 * height_range
            max_error = max(max_error, float(np.max(np.abs(restored - heights[present]))))
            output_min = min(output_min, float(restored.min()))
            output_max = max(output_max, float(restored.max()))
            solid_cells += int(surface.sum())
            tiles.append(dict(x=x, z=z, raw=f"{name}.raw", mask=mask_name))
    for tile in tiles:
        x, z = tile["x"], tile["z"]
        raw = np.fromfile(OUTPUT / tile["raw"], dtype="<u2").reshape(RESOLUTION, RESOLUTION)
        if x:
            left = np.fromfile(OUTPUT / f"tile_{x - 1}_{z}.raw", dtype="<u2").reshape(raw.shape)
            assert np.array_equal(left[:, -1], raw[:, 0]), "East/west seam"
        if z:
            south = np.fromfile(OUTPUT / f"tile_{x}_{z - 1}.raw", dtype="<u2").reshape(raw.shape)
            assert np.array_equal(south[-1], raw[0]), "North/south seam"
    assert max_error <= height_range / 65535 / 2 + 1e-8
    with SOURCE.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest()
    metadata = dict(
        source=str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
        sourceSha256=checksum,
        sourceWidth=width, sourceHeight=height, pixelSizeX=sx, pixelSizeY=sy,
        projection=tags[34737].rstrip("|"), projectionMethod="Equirectangular",
        sphereRadius=keys[2057], standardParallel=keys[3078],
        centralMeridian=keys[3088], latitudeOfOrigin=keys[3089],
        falseEasting=keys[3082], falseNorthing=keys[3083],
        west=west, south=north - length_m, east=west + width_m, north=north,
        noData=nodata, noDataPixels=int((~valid).sum()),
        widthMeters=width_m, lengthMeters=length_m,
        baseElevation=minimum, heightRange=height_range,
        resolution=RESOLUTION, columns=columns, rows=rows,
        tileWidth=width_m / columns, tileLength=length_m / rows,
        sampleSpacingX=width_m / (len(xs) - 1), sampleSpacingZ=length_m / (len(ys) - 1),
        axes="X east, Y source elevation, Z north; X/Z origin at southwest raster corner",
        encoding="Little-endian uint16; rows south to north; columns west to east",
        resampling="Bilinear at pixel centers; outer half-pixels clamped to edge samples",
        missingData="Hide any quad touching a missing sample; mask 1 = surface, 0 = hole",
        maxQuantizationError=max_error, convertedMin=output_min, convertedMax=output_max,
        surfaceCells=solid_cells, tiles=tiles,
    )
    (OUTPUT / "terrain.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{width} x {height}, {sx:g} x {sy:g} m/pixel; {metadata['projection']}")
    print(f"Full footprint: {width_m:g} x {length_m:g} m; elevation {minimum:.6f} to {maximum:.6f} m")
    print(f"No-data: {nodata:g}, {metadata['noDataPixels']} pixels, masked as terrain holes")
    print(f"Wrote {columns} x {rows} tiles; all shared edges match exactly")
    print(f"Maximum 16-bit quantization error: {max_error:.6f} m")
    print(f"Converted elevation range: {output_min:.6f} to {output_max:.6f} m")


if __name__ == "__main__":
    main()
