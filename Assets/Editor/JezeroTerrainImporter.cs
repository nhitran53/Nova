using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

public static class JezeroTerrainImporter
{
    const string Folder = "Assets/JezeroTerrain/Generated";
    const string PrefabPath = Folder + "/Jezero Terrain.prefab";

    [MenuItem("Tools/Jezero/Update Georeference")]
    public static void UpdateGeoreference()
    {
        var map = JsonUtility.FromJson<Map>(File.ReadAllText(Folder + "/terrain.json"));
        ValidateFiles(map);
        if (File.Exists(DetailFolder + "/terrain.json")) ReadHiRise(map);
        var root = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            foreach (var tile in map.tiles)
            {
                var terrain = root.transform.Find($"Tile {tile.x}, {tile.z}");
                var expected = new Vector3(tile.x * map.tileWidth, map.baseElevation, tile.z * map.tileLength);
                if (terrain == null || terrain.localPosition != expected || terrain.localRotation != Quaternion.identity ||
                    terrain.localScale != Vector3.one || terrain.GetComponent<Terrain>().terrainData.size !=
                    new Vector3(map.tileWidth, map.heightRange, map.tileLength))
                    throw new InvalidDataException("Terrain geometry does not match its georeferencing.");
            }
            ApplyGeoreference(root);
            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        }
        finally { PrefabUtility.UnloadPrefabContents(root); }
        Debug.Log("Jezero georeferencing updated. The scene placement was preserved.");
    }

    static void ApplyGeoreference(GameObject root)
    {
        string json = File.ReadAllText(Folder + "/terrain.json");
        var metadata = JsonUtility.FromJson<JezeroGeoreference.Metadata>(json);
        if (metadata == null || !metadata.Valid)
            throw new InvalidDataException("Unsupported terrain coordinate metadata.");
        var reference = root.GetComponent<JezeroGeoreference>();
        if (reference == null) reference = root.AddComponent<JezeroGeoreference>();
        reference.metadata = metadata;
    }

    [Serializable]
    class Tile
    {
        public int x, z;
        public string raw, mask;
    }

    [Serializable]
    class Map
    {
        public int resolution, columns, rows;
        public float widthMeters, lengthMeters, tileWidth, tileLength;
        public float baseElevation, heightRange;
        public Tile[] tiles;
    }

    [Serializable]
    class HiRiseMap
    {
        public int version, resolution, ctxTileX, ctxTileZ, ctxCellX, ctxCellZ, ctxCells;
        public float x, z, widthMeters, lengthMeters, baseElevation, heightRange, collarMeters;
        public string raw, texture, ctxRaw, ctxManifestSha256, ctxRawSha256;
    }

    const string DetailFolder = Folder + "/HiRISE";

    [MenuItem("Tools/Jezero/Import Terrain")]
    public static void Import()
    {
        Import(false);
    }

    [MenuItem("Tools/Jezero/Import HiRISE Test Area")]
    public static void ImportHiRise()
    {
        Import(true);
    }

    static void Import(bool includeHiRise)
    {
        string manifest = Folder + "/terrain.json";
        if (!File.Exists(manifest))
            throw new FileNotFoundException("Run Tools/convert_jezero.py with NumPy and Pillow first.", manifest);
        var map = JsonUtility.FromJson<Map>(File.ReadAllText(manifest));
        ValidateFiles(map);
        var detail = includeHiRise ? ReadHiRise(map) : null;
        AssetDatabase.Refresh();
        var shader = Shader.Find("Universal Render Pipeline/Terrain/Lit");
        if (shader == null)
            throw new InvalidOperationException("The URP Terrain/Lit shader is missing.");

        var material = AssetDatabase.LoadAssetAtPath<Material>(Folder + "/Terrain.mat");
        if (material == null)
        {
            material = new Material(shader);
            AssetDatabase.CreateAsset(material, Folder + "/Terrain.mat");
        }
        var color = AssetDatabase.LoadAssetAtPath<Texture2D>(Folder + "/GroundColor.asset");
        if (color == null)
        {
            color = new Texture2D(1, 1, TextureFormat.RGBA32, false);
            color.SetPixel(0, 0, new Color(0.48f, 0.34f, 0.24f));
            color.Apply();
            AssetDatabase.CreateAsset(color, Folder + "/GroundColor.asset");
        }
        var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(Folder + "/Ground.terrainlayer");
        if (layer == null)
        {
            layer = new TerrainLayer { diffuseTexture = color, tileSize = new Vector2(1000, 1000) };
            AssetDatabase.CreateAsset(layer, Folder + "/Ground.terrainlayer");
        }

        var root = new GameObject("Jezero Terrain");
        var terrains = new Terrain[map.columns, map.rows];
        try
        {
            ApplyGeoreference(root);
            foreach (var tile in map.tiles)
            {
                var heights = ReadHeights(tile, map.resolution);
                string path = $"{Folder}/Tile_{tile.x}_{tile.z}.asset";
                var data = AssetDatabase.LoadAssetAtPath<TerrainData>(path);
                if (data == null)
                {
                    data = new TerrainData();
                    AssetDatabase.CreateAsset(data, path);
                }
                data.heightmapResolution = map.resolution;
                data.size = new Vector3(map.tileWidth, map.heightRange, map.tileLength);
                data.SetHeights(0, 0, heights);
                data.enableHolesTextureCompression = false;
                var surface = ReadSurface(tile, map.resolution - 1);
                data.SetHoles(0, 0, surface);
                data.terrainLayers = new[] { layer };
                data.alphamapResolution = 16;
                var coverage = new float[16, 16, 1];
                for (int z = 0; z < 16; z++)
                    for (int x = 0; x < 16; x++)
                        coverage[z, x, 0] = 1;
                data.SetAlphamaps(0, 0, coverage);
                EditorUtility.SetDirty(data);

                var obj = Terrain.CreateTerrainGameObject(data);
                obj.name = $"Tile {tile.x}, {tile.z}";
                obj.transform.SetParent(root.transform, false);
                obj.transform.localPosition = new Vector3(tile.x * map.tileWidth,
                    map.baseElevation, tile.z * map.tileLength);
                var terrain = obj.GetComponent<Terrain>();
                terrain.materialTemplate = material;
                terrain.drawInstanced = true;
                terrain.heightmapPixelError = 3;
                terrain.basemapDistance = 120000;
                terrain.groupingID = 2020;
                terrain.allowAutoConnect = true;
                terrains[tile.x, tile.z] = terrain;
            }

            for (int z = 0; z < map.rows; z++)
                for (int x = 0; x < map.columns; x++)
                    terrains[x, z].SetNeighbors(x > 0 ? terrains[x - 1, z] : null,
                        z + 1 < map.rows ? terrains[x, z + 1] : null,
                        x + 1 < map.columns ? terrains[x + 1, z] : null,
                        z > 0 ? terrains[x, z - 1] : null);

            if (detail != null)
            {
                AddHiRise(root, terrains[detail.ctxTileX, detail.ctxTileZ], map, detail);
                JezeroSurfaceDetail.Apply(root);
            }

            AssetDatabase.SaveAssets();
            PrefabUtility.SaveAsPrefabAsset(root, PrefabPath);
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(root);
        }

        var saved = PrefabUtility.LoadPrefabContents(PrefabPath);
        try
        {
            var tiles = saved.GetComponentsInChildren<Terrain>().Where(t => t.groupingID == 2020).ToArray();
            if (tiles.Length != map.tiles.Length)
                throw new InvalidDataException("Saved terrain tile count is wrong.");
            foreach (var terrain in tiles)
                if (terrain.terrainData == null || terrain.materialTemplate == null ||
                    terrain.GetComponent<TerrainCollider>().terrainData != terrain.terrainData)
                    throw new InvalidDataException("A saved terrain reference is missing.");
            foreach (var tile in map.tiles)
            {
                var transform = saved.transform.Find($"Tile {tile.x}, {tile.z}");
                var expectedPosition = new Vector3(tile.x * map.tileWidth,
                    map.baseElevation, tile.z * map.tileLength);
                var expectedSize = new Vector3(map.tileWidth, map.heightRange, map.tileLength);
                if (transform == null || transform.localPosition != expectedPosition ||
                    transform.localScale != Vector3.one || transform.localRotation != Quaternion.identity)
                    throw new InvalidDataException("A saved tile is misplaced.");
                var data = transform.GetComponent<Terrain>().terrainData;
                if (data.size != expectedSize)
                    throw new InvalidDataException("A saved tile has the wrong dimensions.");
                var surface = ReadSurface(tile, map.resolution - 1);
                if (detail != null && tile.x == detail.ctxTileX && tile.z == detail.ctxTileZ) CutHole(surface, detail);
                VerifySamples(data, ReadHeights(tile, map.resolution), surface);
            }
            if (detail != null) VerifyHiRise(saved, map, detail);
        }
        finally
        {
            PrefabUtility.UnloadPrefabContents(saved);
        }
        Debug.Log($"Jezero terrain imported and verified: {map.columns} x {map.rows} tiles, " +
            $"{map.widthMeters} x {map.lengthMeters} meters, elevation " +
            $"{map.baseElevation} to {map.baseElevation + map.heightRange} meters. " +
            $"Open {PrefabPath} to inspect it.");
        if (!Application.isBatchMode)
            AssetDatabase.OpenAsset(AssetDatabase.LoadAssetAtPath<GameObject>(PrefabPath));
    }

    static string Sha256(string path)
    {
        using var stream = File.OpenRead(path);
        using var hash = SHA256.Create();
        return BitConverter.ToString(hash.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
    }

    static HiRiseMap ReadHiRise(Map ctx)
    {
        string path = DetailFolder + "/terrain.json";
        if (!File.Exists(path)) throw new FileNotFoundException("Run python Tools/convert_hirise.py first.", path);
        var map = JsonUtility.FromJson<HiRiseMap>(File.ReadAllText(path));
        if (map == null || map.version != 1 || map.resolution != 513 || map.ctxCells != 32 ||
            map.ctxTileX < 0 || map.ctxTileX >= ctx.columns || map.ctxTileZ < 0 || map.ctxTileZ >= ctx.rows ||
            map.ctxCellX <= 0 || map.ctxCellZ <= 0 || map.ctxCellX + map.ctxCells >= 1024 ||
            map.ctxCellZ + map.ctxCells >= 1024 || !(map.heightRange > 0) || !(map.collarMeters > 0) ||
            map.raw != "heights.raw" || map.texture != "ortho.png" ||
            map.ctxRaw != $"tile_{map.ctxTileX}_{map.ctxTileZ}.raw")
            throw new InvalidDataException("Invalid HiRISE test area manifest.");
        float spacingX = ctx.tileWidth / 1024, spacingZ = ctx.tileLength / 1024;
        if (Mathf.Abs(map.x - (map.ctxTileX * ctx.tileWidth + map.ctxCellX * spacingX)) > .01f ||
            Mathf.Abs(map.z - (map.ctxTileZ * ctx.tileLength + map.ctxCellZ * spacingZ)) > .01f ||
            Mathf.Abs(map.widthMeters - map.ctxCells * spacingX) > .01f ||
            Mathf.Abs(map.lengthMeters - map.ctxCells * spacingZ) > .01f ||
            Sha256(Folder + "/terrain.json") != map.ctxManifestSha256 ||
            Sha256(Folder + "/" + map.ctxRaw) != map.ctxRawSha256)
            throw new InvalidDataException("HiRISE mapping does not match the current CTX data; rerun the converter.");
        if (new FileInfo(DetailFolder + "/" + map.raw).Length != map.resolution * map.resolution * 2L ||
            !File.Exists(DetailFolder + "/" + map.texture))
            throw new InvalidDataException("HiRISE heightmap or image missing/invalid.");
        return map;
    }

    static void CutHole(bool[,] surface, HiRiseMap map)
    {
        for (int z = map.ctxCellZ; z < map.ctxCellZ + map.ctxCells; z++)
            for (int x = map.ctxCellX; x < map.ctxCellX + map.ctxCells; x++) surface[z, x] = false;
    }

    static float CollarWeight(int x, int z, HiRiseMap map)
    {
        float u = x / (map.resolution - 1f), v = z / (map.resolution - 1f);
        float edge = Mathf.Min(u * map.widthMeters, (1 - u) * map.widthMeters,
            v * map.lengthMeters, (1 - v) * map.lengthMeters);
        return Mathf.SmoothStep(0, 1, Mathf.Clamp01(edge / map.collarMeters));
    }

    static float Sample(float[,] values, float x, float z)
    {
        int ix = Mathf.FloorToInt(x), iz = Mathf.FloorToInt(z);
        return Mathf.Lerp(Mathf.Lerp(values[iz, ix], values[iz, ix + 1], x - ix),
            Mathf.Lerp(values[iz + 1, ix], values[iz + 1, ix + 1], x - ix), z - iz);
    }

    static void AddHiRise(GameObject root, Terrain ctxTerrain, Map ctx, HiRiseMap map)
    {
        var original = ctxTerrain.terrainData;
        var originalSurface = original.GetHoles(0, 0, 1024, 1024);
        for (int z = map.ctxCellZ - 1; z <= map.ctxCellZ + map.ctxCells; z++)
            for (int x = map.ctxCellX - 1; x <= map.ctxCellX + map.ctxCells; x++)
                if (!originalSurface[z, x]) throw new InvalidDataException("CTX is missing beneath the patch.");

        string texturePath = DetailFolder + "/" + map.texture;
        var importer = (TextureImporter)AssetImporter.GetAtPath(texturePath);
        importer.textureType = TextureImporterType.Default;
        importer.sRGBTexture = true;
        importer.mipmapEnabled = true;
        importer.isReadable = false;
        importer.wrapMode = TextureWrapMode.Clamp;
        importer.filterMode = FilterMode.Trilinear;
        importer.anisoLevel = 2;
        importer.maxTextureSize = 2048;
        importer.textureCompression = TextureImporterCompression.Compressed;
        importer.SetPlatformTextureSettings(new TextureImporterPlatformSettings
        {
            name = "Android", overridden = true, maxTextureSize = 2048,
            format = TextureImporterFormat.ASTC_6x6
        });
        importer.SaveAndReimport();
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(texturePath);
        if (texture == null || texture.width != 2048 || texture.height != 2048)
            throw new InvalidDataException("HiRISE texture import failed.");

        string layerPath = DetailFolder + "/HiRISE.terrainlayer";
        var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(layerPath);
        if (layer == null)
        {
            layer = new TerrainLayer();
            AssetDatabase.CreateAsset(layer, layerPath);
        }
        layer.diffuseTexture = texture;
        layer.tileSize = new Vector2(map.widthMeters, map.lengthMeters);
        layer.tileOffset = Vector2.zero;
        layer.metallic = layer.smoothness = 0;
        EditorUtility.SetDirty(layer);

        var heights = new float[map.resolution, map.resolution];
        var coarse = ReadHeights(new Tile { raw = map.ctxRaw }, 1025);
        using (var reader = new BinaryReader(File.OpenRead(DetailFolder + "/" + map.raw)))
            for (int z = 0; z < map.resolution; z++)
                for (int x = 0; x < map.resolution; x++)
                {
                    float h = reader.ReadUInt16() / 65535f;
                    float cx = map.ctxCellX + x / (map.resolution - 1f) * map.ctxCells;
                    float cz = map.ctxCellZ + z / (map.resolution - 1f) * map.ctxCells;
                    // Match the stored CTX heights along the patch edge to avoid seams.
                    float sourceY = ctx.baseElevation + Sample(coarse, cx, cz) * ctx.heightRange;
                    float unityY = ctx.baseElevation + original.GetInterpolatedHeight(cx / 1024, cz / 1024);
                    heights[z, x] = h + (unityY - sourceY) * (1 - CollarWeight(x, z, map)) / map.heightRange;
                    if (heights[z, x] < 0 || heights[z, x] > 1 || float.IsNaN(heights[z, x]))
                        throw new InvalidDataException("HiRISE height is outside its declared elevation range.");
                }

        string dataPath = DetailFolder + "/HiRISE.asset";
        var data = AssetDatabase.LoadAssetAtPath<TerrainData>(dataPath);
        if (data == null)
        {
            data = new TerrainData();
            AssetDatabase.CreateAsset(data, dataPath);
        }
        data.heightmapResolution = map.resolution;
        data.size = new Vector3(map.widthMeters, map.heightRange, map.lengthMeters);
        data.SetHeights(0, 0, heights);
        data.terrainLayers = new[] { layer };
        data.alphamapResolution = 16;
        data.baseMapResolution = 1024;
        var coverage = new float[16, 16, 1];
        for (int z = 0; z < 16; z++) for (int x = 0; x < 16; x++) coverage[z, x, 0] = 1;
        data.SetAlphamaps(0, 0, coverage);
        EditorUtility.SetDirty(data);

        string cutPath = DetailFolder + "/CTX_With_Test_Area.asset";
        var cut = AssetDatabase.LoadAssetAtPath<TerrainData>(cutPath);
        if (cut == null)
        {
            cut = UnityEngine.Object.Instantiate(original);
            AssetDatabase.CreateAsset(cut, cutPath);
        }
        else EditorUtility.CopySerialized(original, cut);
        CutHole(originalSurface, map);
        cut.SetHoles(0, 0, originalSurface);
        EditorUtility.SetDirty(cut);
        ctxTerrain.terrainData = cut;
        ctxTerrain.GetComponent<TerrainCollider>().terrainData = cut;

        var obj = Terrain.CreateTerrainGameObject(data);
        obj.name = "HiRISE Test Area";
        obj.transform.SetParent(root.transform, false);
        obj.transform.localPosition = new Vector3(map.x, map.baseElevation, map.z);
        var terrain = obj.GetComponent<Terrain>();
        terrain.materialTemplate = ctxTerrain.materialTemplate;
        terrain.drawInstanced = true;
        terrain.heightmapPixelError = 3;
        terrain.basemapDistance = 750;
        terrain.drawTreesAndFoliage = false;
        terrain.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        terrain.groupingID = 2021;
        terrain.allowAutoConnect = false;
    }

    static void VerifyHiRise(GameObject root, Map ctx, HiRiseMap map)
    {
        var transform = root.transform.Find("HiRISE Test Area");
        var terrain = transform == null ? null : transform.GetComponent<Terrain>();
        if (terrain == null || terrain.GetComponent<TerrainCollider>().terrainData != terrain.terrainData ||
            transform.localPosition != new Vector3(map.x, map.baseElevation, map.z) ||
            terrain.terrainData.size != new Vector3(map.widthMeters, map.heightRange, map.lengthMeters) ||
            terrain.terrainData.terrainLayers[0].diffuseTexture == null)
            throw new InvalidDataException("HiRISE terrain references or placement invalid.");
        var coarse = root.transform.Find($"Tile {map.ctxTileX}, {map.ctxTileZ}").GetComponent<Terrain>().terrainData;
        var heights = terrain.terrainData.GetHeights(0, 0, map.resolution, map.resolution);
        float maxSeam = 0;
        for (int z = 0; z < map.resolution; z++)
            for (int x = 0; x < map.resolution; x++)
                if (x == 0 || z == 0 || x == map.resolution - 1 || z == map.resolution - 1)
                {
                    float cx = map.ctxCellX + x / (map.resolution - 1f) * map.ctxCells;
                    float cz = map.ctxCellZ + z / (map.resolution - 1f) * map.ctxCells;
                    float actual = map.baseElevation + heights[z, x] * map.heightRange;
                    float expected = ctx.baseElevation + coarse.GetInterpolatedHeight(cx / 1024, cz / 1024);
                    maxSeam = Mathf.Max(maxSeam, Mathf.Abs(actual - expected));
                }
        if (maxSeam > .002f) throw new InvalidDataException($"HiRISE edge mismatch: {maxSeam} m");
        Debug.Log($"HiRISE test area: {map.widthMeters} x {map.lengthMeters} m, max boundary mismatch {maxSeam} m. " +
            "Original 25 CTX heightfields preserved; one copied tile has the matching render/collider hole.");
    }

    static void ValidateFiles(Map map)
    {
        if (map == null || map.resolution != 1025 || map.columns <= 0 || map.rows <= 0 ||
            map.tiles == null || map.tiles.Length != map.columns * map.rows ||
            !(map.heightRange > 0) || !(map.tileWidth > 0) || !(map.tileLength > 0) ||
            Mathf.Abs(map.tileWidth * map.columns - map.widthMeters) > 0.01f ||
            Mathf.Abs(map.tileLength * map.rows - map.lengthMeters) > 0.01f)
            throw new InvalidDataException("Invalid Jezero terrain metadata.");
        var occupied = new bool[map.columns, map.rows];
        foreach (var tile in map.tiles)
        {
            if (tile.x < 0 || tile.x >= map.columns || tile.z < 0 || tile.z >= map.rows ||
                occupied[tile.x, tile.z])
                throw new InvalidDataException("Invalid or duplicate tile position.");
            occupied[tile.x, tile.z] = true;
            if (new FileInfo(Folder + "/" + tile.raw).Length != map.resolution * map.resolution * 2L ||
                (!string.IsNullOrEmpty(tile.mask) && new FileInfo(Folder + "/" + tile.mask).Length !=
                    (map.resolution - 1) * (map.resolution - 1)))
                throw new InvalidDataException("A heightmap or surface mask has the wrong size.");
        }
    }

    static float[,] ReadHeights(Tile tile, int resolution)
    {
        var heights = new float[resolution, resolution];
        using var reader = new BinaryReader(File.OpenRead(Folder + "/" + tile.raw));
        for (int z = 0; z < resolution; z++)
            for (int x = 0; x < resolution; x++)
                heights[z, x] = reader.ReadUInt16() / 65535f;
        return heights;
    }

    static bool[,] ReadSurface(Tile tile, int resolution)
    {
        var surface = new bool[resolution, resolution];
        var mask = string.IsNullOrEmpty(tile.mask) ? null : File.ReadAllBytes(Folder + "/" + tile.mask);
        for (int z = 0; z < resolution; z++)
            for (int x = 0; x < resolution; x++)
                surface[z, x] = mask == null || mask[z * resolution + x] != 0;
        return surface;
    }

    static void VerifySamples(TerrainData data, float[,] expected, bool[,] surface)
    {
        int n = data.heightmapResolution;
        var actual = data.GetHeights(0, 0, n, n);
        var holes = data.GetHoles(0, 0, n - 1, n - 1);
        for (int z = 0; z < n; z++)
            for (int x = 0; x < n; x++)
                if (Mathf.Abs(actual[z, x] - expected[z, x]) > 2f / 65535)
                    throw new InvalidDataException("Unity changed an imported height sample.");
        for (int z = 0; z < n - 1; z++)
            for (int x = 0; x < n - 1; x++)
                if (holes[z, x] != surface[z, x])
                    throw new InvalidDataException("Unity changed the no-data mask.");
    }
}
