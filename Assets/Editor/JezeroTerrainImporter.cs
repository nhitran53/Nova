using System;
using System.IO;
using UnityEditor;
using UnityEngine;

public static class JezeroTerrainImporter
{
    const string Folder = "Assets/JezeroTerrain/Generated";
    const string PrefabPath = Folder + "/Jezero Terrain.prefab";

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

    [MenuItem("Tools/Jezero/Import Terrain")]
    public static void Import()
    {
        string manifest = Folder + "/terrain.json";
        if (!File.Exists(manifest))
            throw new FileNotFoundException("Run Tools/convert_jezero.py with NumPy and Pillow first.", manifest);
        var map = JsonUtility.FromJson<Map>(File.ReadAllText(manifest));
        ValidateFiles(map);
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
            var tiles = saved.GetComponentsInChildren<Terrain>();
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
                VerifySamples(data, ReadHeights(tile, map.resolution), ReadSurface(tile, map.resolution - 1));
            }
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
