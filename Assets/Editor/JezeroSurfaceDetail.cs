using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class JezeroSurfaceDetail
{
    const string Folder = "Assets/JezeroTerrain/Generated/SurfaceDetail";
    const string Prefab = "Assets/JezeroTerrain/Generated/Jezero Terrain.prefab";
    const int TextureSize = 512;

    [MenuItem("Tools/Jezero/Build Surface Detail")]
    public static void Build()
    {
        if (EditorApplication.isPlaying) throw new InvalidOperationException("Stop Play mode before rebuilding surface detail.");
        var asset = AssetDatabase.LoadAssetAtPath<GameObject>(Prefab);
        if (asset == null) throw new FileNotFoundException("Import the Jezero HiRISE test area first.", Prefab);
        var root = PrefabUtility.LoadPrefabContents(Prefab);
        try
        {
            Apply(root);
            AssetDatabase.SaveAssets();
            PrefabUtility.SaveAsPrefabAsset(root, Prefab);
        }
        finally { PrefabUtility.UnloadPrefabContents(root); }
        foreach (var field in UnityEngine.Object.FindObjectsByType<JezeroRockField>(FindObjectsInactive.Include, FindObjectsSortMode.None))
            if (field.gameObject.scene == SceneManager.GetActiveScene()) field.Rebuild();
        SceneView.RepaintAll();
    }

    public static void Apply(GameObject root)
    {
        var terrain = root.GetComponentsInChildren<Terrain>(true).SingleOrDefault(t => t.name == "HiRISE Test Area");
        if (terrain == null) throw new InvalidOperationException("The generated prefab has no HiRISE Test Area.");
        Directory.CreateDirectory(Folder);
        AssetDatabase.Refresh();
        MakeGroundTextures("Dust", 5.7f, 171, false);
        MakeGroundTextures("Gravel", 3.1f, 643, true);
        var dust = Layer("Dust", 5.7f, Vector2.zero);
        var gravel = Layer("Gravel", 3.1f, new Vector2(1.13f, .47f));
        var data = terrain.terrainData;
        var imagery = AssetDatabase.LoadAssetAtPath<TerrainLayer>("Assets/JezeroTerrain/Generated/HiRISE/HiRISE.terrainlayer");
        if (imagery == null || imagery.diffuseTexture == null) throw new InvalidOperationException("HiRISE imagery is missing.");
        data.terrainLayers = new[] { imagery, dust, gravel };
        data.alphamapResolution = 128;
        var coverage = new float[128,128,3];
        for (int z = 0; z < 128; z++) for (int x = 0; x < 128; x++)
        {
            float u = x / 127f, v = z / 127f;
            float mx = u * data.size.x, mz = v * data.size.z;
            float edge = Mathf.Min(mx, mz, data.size.x-mx, data.size.z-mz);
            float fade = Mathf.SmoothStep(0,1,Mathf.Clamp01(edge/64));
            float patch = Mathf.PerlinNoise(mx*.025f+3.1f,mz*.025f+7.3f);
            float slope = data.GetSteepness(u,v);
            float gravelWeight = fade * (.06f + .2f * patch * (1-Mathf.Clamp01(slope/40)));
            float dustWeight = fade * (.16f + .06f * (1-patch));
            coverage[z,x,0] = 1-dustWeight-gravelWeight;
            coverage[z,x,1] = dustWeight;
            coverage[z,x,2] = gravelWeight;
        }
        data.SetAlphamaps(0,0,coverage);
        EditorUtility.SetDirty(data);

        var sceneTerrain = SceneTerrains().FirstOrDefault(t => t.name == "HiRISE Test Area");
        var sourceSettings = sceneTerrain != null ? sceneTerrain.GetComponent<JezeroRockField>() : null;
        if (sourceSettings == null)
        {
            var old = AssetDatabase.LoadAssetAtPath<GameObject>(Prefab);
            if (old != null) sourceSettings = old.GetComponentInChildren<JezeroRockField>(true);
        }
        var field = terrain.GetComponent<JezeroRockField>();
        bool created = field == null;
        if (created) field = terrain.gameObject.AddComponent<JezeroRockField>();
        if (sourceSettings != null && sourceSettings != field) EditorUtility.CopySerialized(sourceSettings,field);
        field.terrain = terrain;
        if (created && sourceSettings == null)
        {
            var player = UnityEngine.Object.FindObjectsByType<CharacterController>(FindObjectsInactive.Include,FindObjectsSortMode.None)
                .FirstOrDefault(p => p.gameObject.scene == SceneManager.GetActiveScene() && p.GetComponentInParent<ExplorerRigs>(true) != null);
            var local = sceneTerrain != null && player != null ? sceneTerrain.transform.InverseTransformPoint(player.transform.position) : data.size * .5f;
            field.center = new Vector2(local.x,local.z);
        }
        field.nearMeshes = new Mesh[8];
        field.farMeshes = new Mesh[8];
        for (int i = 0; i < 8; i++)
        {
            field.nearMeshes[i] = SaveMesh(RockMesh(i,2,"Near"+i));
            field.farMeshes[i] = SaveMesh(RockMesh(i,1,"Far"+i));
        }
        var usedMeshes = new HashSet<Mesh>(field.nearMeshes.Concat(field.farMeshes));
        foreach (var mesh in AssetDatabase.LoadAllAssetsAtPath(Folder+"/Rocks.asset").OfType<Mesh>())
            if (!usedMeshes.Contains(mesh) && !AssetDatabase.IsMainAsset(mesh)) UnityEngine.Object.DestroyImmediate(mesh,true);
        field.materials = new Material[3];
        var colors = new[] { new Color(.7f,.73f,.76f),new Color(.85f,.82f,.76f),new Color(.58f,.62f,.68f) };
        for (int i = 0; i < 3; i++)
        {
            string path = $"{Folder}/Rock{i}.mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Lit");
                if (shader == null) throw new InvalidOperationException("URP Lit shader is missing.");
                material = new Material(shader);
                AssetDatabase.CreateAsset(material,path);
            }
            material.SetTexture("_BaseMap",gravel.diffuseTexture);
            material.SetTexture("_BumpMap",dust.normalMapTexture);
            material.SetColor("_BaseColor",colors[i]);
            material.SetFloat("_Smoothness",.08f);
            material.SetFloat("_BumpScale",.6f);
            material.EnableKeyword("_NORMALMAP");
            material.enableInstancing = true;
            EditorUtility.SetDirty(material);
            field.materials[i] = material;
        }
        field.Rebuild();
        var oldColliders = terrain.transform.Find("Rock Colliders");
        if (oldColliders != null) UnityEngine.Object.DestroyImmediate(oldColliders.gameObject);
        var colliders = new GameObject("Rock Colliders");
        colliders.transform.SetParent(terrain.transform,false);
        int count = 0;
        foreach (var rock in field.Rocks)
        {
            if (rock.kind != 2) continue;
            var obj = new GameObject("Boulder "+count++);
            obj.transform.SetParent(colliders.transform,false);
            obj.transform.localPosition = rock.position;
            obj.transform.localRotation = rock.rotation;
            obj.transform.localScale = rock.scale;
            var collider = obj.AddComponent<MeshCollider>();
            collider.sharedMesh = field.farMeshes[rock.variant];
            collider.convex = true;
        }
        EditorUtility.SetDirty(field);
        Debug.Log($"Jezero surface detail: {field.Rocks.Count} instanced rocks within {field.radius} m, {count} boulder colliders. Heightfield and imagery unchanged.");
    }

    static IEnumerable<Terrain> SceneTerrains() => UnityEngine.Object.FindObjectsByType<Terrain>(FindObjectsInactive.Include,FindObjectsSortMode.None)
        .Where(t => t.gameObject.scene == SceneManager.GetActiveScene());

    static TerrainLayer Layer(string name, float size, Vector2 offset)
    {
        string path = $"{Folder}/{name}.terrainlayer";
        var layer = AssetDatabase.LoadAssetAtPath<TerrainLayer>(path);
        if (layer == null) { layer = new TerrainLayer(); AssetDatabase.CreateAsset(layer,path); }
        layer.diffuseTexture = AssetDatabase.LoadAssetAtPath<Texture2D>($"{Folder}/{name}.png");
        layer.normalMapTexture = AssetDatabase.LoadAssetAtPath<Texture2D>($"{Folder}/{name}Normal.png");
        layer.tileSize = Vector2.one * size;
        layer.tileOffset = offset;
        layer.normalScale = 1.4f;
        layer.smoothness = layer.metallic = 0;
        EditorUtility.SetDirty(layer);
        return layer;
    }

    static float Hash(int x, int y, int seed)
    {
        uint h = unchecked((uint)x*374761393u+(uint)y*668265263u+(uint)seed*2246822519u);
        h = (h^(h>>13))*1274126177u;
        return (h^(h>>16)) / (float)uint.MaxValue;
    }

    static int Wrap(int value, int period) => (value%period+period)%period;
    static float Noise(float u, float v, int cells, int seed)
    {
        float x = u*cells, y = v*cells;
        int ix = Mathf.FloorToInt(x), iy = Mathf.FloorToInt(y);
        float tx = Mathf.SmoothStep(0,1,x-ix), ty = Mathf.SmoothStep(0,1,y-iy);
        return Mathf.Lerp(Mathf.Lerp(Hash(Wrap(ix,cells),Wrap(iy,cells),seed),Hash(Wrap(ix+1,cells),Wrap(iy,cells),seed),tx),
            Mathf.Lerp(Hash(Wrap(ix,cells),Wrap(iy+1,cells),seed),Hash(Wrap(ix+1,cells),Wrap(iy+1,cells),seed),tx),ty);
    }

    static float Pebble(float u, float v, int seed)
    {
        float x = u*40, y = v*40, nearest = 10;
        int ix = Mathf.FloorToInt(x), iy = Mathf.FloorToInt(y);
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
        {
            int cx = ix+dx, cy = iy+dy;
            float px = cx+.1f+.8f*Hash(Wrap(cx,40),Wrap(cy,40),seed);
            float py = cy+.1f+.8f*Hash(Wrap(cx,40),Wrap(cy,40),seed+1);
            float radius = .2f+.22f*Hash(Wrap(cx,40),Wrap(cy,40),seed+2);
            nearest = Mathf.Min(nearest,new Vector2(x-px,y-py).magnitude/radius);
        }
        return Mathf.Pow(Mathf.Clamp01(1-nearest),.7f);
    }

    static void MakeGroundTextures(string name, float meters, int seed, bool gravel)
    {
        int n = TextureSize;
        var heights = new float[n*n];
        var colors = new Color[n*n];
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
        {
            float u = x/(float)n, v = y/(float)n;
            float wu = u + .03f*Noise(u,v,11,seed+11), wv = v + .03f*Noise(u,v,13,seed+12);
            float grain = .7f*Noise(wu,wv,113,seed)+.3f*Noise(wu,wv,197,seed+6);
            float broad = Noise(u,v,7,seed+3);
            float pebble = gravel ? Pebble(u,v,seed+5) : 0;
            heights[y*n+x] = .002f*grain + .001f*Noise(u,v,37,seed+4) + .014f*pebble;
            float shade = gravel ? .55f+.35f*broad+.3f*grain+.6f*pebble : .7f+.25f*broad+.3f*grain;
            colors[y*n+x] = new Color(.43f*shade,.32f*shade,.24f*shade,1);
        }
        SaveTexture(name,colors,false);
        var normals = new Color[n*n];
        float step = meters/n;
        for (int y = 0; y < n; y++) for (int x = 0; x < n; x++)
        {
            float dx = (heights[y*n+Wrap(x-1,n)]-heights[y*n+Wrap(x+1,n)])/(2*step);
            float dy = (heights[Wrap(y-1,n)*n+x]-heights[Wrap(y+1,n)*n+x])/(2*step);
            var normal = new Vector3(dx,dy,1).normalized;
            normals[y*n+x] = new Color(normal.x*.5f+.5f,normal.y*.5f+.5f,normal.z*.5f+.5f,1);
        }
        SaveTexture(name+"Normal",normals,true);
    }

    static void SaveTexture(string name, Color[] pixels, bool normal)
    {
        string path = $"{Folder}/{name}.png";
        var texture = new Texture2D(TextureSize,TextureSize,TextureFormat.RGBA32,false,normal);
        texture.SetPixels(pixels); texture.Apply();
        File.WriteAllBytes(path,texture.EncodeToPNG());
        UnityEngine.Object.DestroyImmediate(texture);
        AssetDatabase.ImportAsset(path,ImportAssetOptions.ForceSynchronousImport);
        var importer = (TextureImporter)AssetImporter.GetAtPath(path);
        importer.textureType = normal ? TextureImporterType.NormalMap : TextureImporterType.Default;
        importer.sRGBTexture = !normal;
        importer.mipmapEnabled = true;
        importer.isReadable = false;
        importer.wrapMode = TextureWrapMode.Repeat;
        importer.filterMode = FilterMode.Trilinear;
        importer.anisoLevel = 4;
        importer.maxTextureSize = TextureSize;
        importer.SetPlatformTextureSettings(new TextureImporterPlatformSettings { name="Android",overridden=true,
            maxTextureSize=TextureSize,format=normal ? TextureImporterFormat.ASTC_4x4 : TextureImporterFormat.ASTC_6x6 });
        importer.SaveAndReimport();
    }

    static Mesh RockMesh(int variant, int subdivisions, string name)
    {
        float t = (1+Mathf.Sqrt(5))/2;
        var vertices = new List<Vector3> { new Vector3(-1,t,0),new Vector3(1,t,0),new Vector3(-1,-t,0),new Vector3(1,-t,0),
            new Vector3(0,-1,t),new Vector3(0,1,t),new Vector3(0,-1,-t),new Vector3(0,1,-t),
            new Vector3(t,0,-1),new Vector3(t,0,1),new Vector3(-t,0,-1),new Vector3(-t,0,1) };
        for (int i = 0; i < vertices.Count; i++) vertices[i] = vertices[i].normalized;
        var triangles = new List<int> { 0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,
            3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1 };
        for (int level = 0; level < subdivisions; level++)
        {
            var edges = new Dictionary<long,int>();
            int Mid(int a,int b)
            {
                long key = ((long)Mathf.Min(a,b)<<32)|(uint)Mathf.Max(a,b);
                if (edges.TryGetValue(key,out int found)) return found;
                int index = vertices.Count;
                vertices.Add((vertices[a]+vertices[b]).normalized);
                edges[key] = index;
                return index;
            }
            var next = new List<int>();
            for (int i = 0; i < triangles.Count; i += 3)
            {
                int a=triangles[i],b=triangles[i+1],c=triangles[i+2],ab=Mid(a,b),bc=Mid(b,c),ca=Mid(c,a);
                next.AddRange(new[] { a,ab,ca,b,bc,ab,c,ca,bc,ab,bc,ca });
            }
            triangles = next;
        }
        float phase = variant*1.731f;
        float minY=float.MaxValue,maxY=float.MinValue,maxXZ=0;
        for (int i = 0; i < vertices.Count; i++)
        {
            var v = vertices[i];
            float r = .85f+.14f*Mathf.Sin(Vector3.Dot(v,new Vector3(3,5,7))+phase)+.09f*Mathf.Cos(Vector3.Dot(v,new Vector3(8,-3,4))+phase*2);
            v = Vector3.Scale(v,new Vector3(.85f+variant*.035f,.65f,.95f-variant*.025f))*r;
            v.x += v.y*.17f;
            v.y = Mathf.Clamp(v.y,-.48f,.53f);
            vertices[i]=v;
            minY=Mathf.Min(minY,v.y); maxY=Mathf.Max(maxY,v.y); maxXZ=Mathf.Max(maxXZ,Mathf.Abs(v.x),Mathf.Abs(v.z));
        }
        for (int i = 0; i < vertices.Count; i++)
        {
            var v=vertices[i]; vertices[i]=new Vector3(v.x/(2*maxXZ),(v.y-minY)/(maxY-minY),v.z/(2*maxXZ));
        }
        var flat = new Vector3[triangles.Count]; var uv = new Vector2[flat.Length]; var indices = new int[flat.Length];
        for (int i = 0; i < flat.Length; i++)
        {
            flat[i]=vertices[triangles[i]]; indices[i]=i;
            uv[i]=new Vector2(Mathf.Atan2(flat[i].z,flat[i].x)/(2*Mathf.PI)+.5f,flat[i].y);
        }
        for (int i = 0; i < uv.Length; i += 3)
            if (Mathf.Max(uv[i].x,uv[i+1].x,uv[i+2].x)-Mathf.Min(uv[i].x,uv[i+1].x,uv[i+2].x)>.5f)
                for (int j=0;j<3;j++) if (uv[i+j].x<.5f) uv[i+j].x+=1;
        var mesh = new Mesh { name=name,vertices=flat,triangles=indices,uv=uv };
        mesh.RecalculateNormals(); mesh.RecalculateTangents(); mesh.RecalculateBounds();
        return mesh;
    }

    static Mesh SaveMesh(Mesh generated)
    {
        string path = Folder+"/Rocks.asset";
        var existing = generated.name == "Near0" ? AssetDatabase.LoadAssetAtPath<Mesh>(path) :
            AssetDatabase.LoadAllAssetsAtPath(path).OfType<Mesh>().FirstOrDefault(m => m.name == generated.name);
        if (existing == null)
        {
            if (!File.Exists(path)) AssetDatabase.CreateAsset(generated,path);
            else AssetDatabase.AddObjectToAsset(generated,path);
            return generated;
        }
        existing.Clear(); existing.vertices=generated.vertices; existing.triangles=generated.triangles; existing.uv=generated.uv;
        existing.RecalculateNormals(); existing.RecalculateTangents(); existing.RecalculateBounds();
        UnityEngine.Object.DestroyImmediate(generated);
        EditorUtility.SetDirty(existing);
        return existing;
    }
}
