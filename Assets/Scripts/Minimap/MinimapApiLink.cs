using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

// Optional API link. Leave apiBaseUrl empty to disable it.
// On Quest use the server's LAN address.
// GET /map adds pins. Scene markers take priority over matching API pins.
// POST /position sends the player location for backend hazard checks.
[RequireComponent(typeof(MarsMinimap))]
public class MinimapApiLink : MonoBehaviour
{
    [Tooltip("e.g. http://192.168.1.50:8000  (empty = off). localhost only works in the Editor.")]
    public string apiBaseUrl = "";
    public bool fetchMap = true;
    public float mapIntervalSeconds = 5f;
    public bool postPosition = true;
    public float positionIntervalSeconds = 1f;

    // /map may return a bare array or an object holding the array; JsonUtility ignores fields it doesn't find.
    [Serializable]
    class MapResponse
    {
        public MapPoint[] items;
        public MapPoint[] locations;
        public MapPoint[] markers;
    }

    [Serializable]
    class PositionBody
    {
        public float x_m;
        public float z_m;
        public float heading_deg;
    }

    MarsMinimap minimap;

    void Awake() => minimap = GetComponent<MarsMinimap>();

    void OnEnable()
    {
        StartCoroutine(MapLoop());
        StartCoroutine(PositionLoop());
    }

    void OnDisable() => StopAllCoroutines();

    string Url(string path) => apiBaseUrl.TrimEnd('/') + path;

    IEnumerator MapLoop()
    {
        while (true)
        {
            if (fetchMap && !string.IsNullOrWhiteSpace(apiBaseUrl))
            {
                using (var req = UnityWebRequest.Get(Url("/map")))
                {
                    req.timeout = 5;
                    yield return req.SendWebRequest();
                    if (req.result == UnityWebRequest.Result.Success) ApplyMap(req.downloadHandler.text);
                    else Debug.LogWarning($"[Minimap] GET /map failed: {req.responseCode} {req.error}");
                }
            }
            yield return new WaitForSeconds(Mathf.Max(1f, mapIntervalSeconds));
        }
    }

    void ApplyMap(string json)
    {
        if (json.TrimStart().StartsWith("[")) json = "{\"items\":" + json + "}";   // JsonUtility can't read a top-level array
        MapResponse parsed;
        try { parsed = JsonUtility.FromJson<MapResponse>(json); }
        catch (Exception e) { Debug.LogWarning($"[Minimap] Could not read /map: {e.Message}"); return; }
        if (parsed == null) return;
        // Use the first nonempty array supported by the API response.
        MapPoint[] pins = parsed.items;
        if (pins == null || pins.Length == 0) pins = parsed.locations;
        if (pins == null || pins.Length == 0) pins = parsed.markers;
        minimap.SetExternalPins(pins);
    }

    IEnumerator PositionLoop()
    {
        while (true)
        {
            if (postPosition && !string.IsNullOrWhiteSpace(apiBaseUrl))
            {
                var body = new PositionBody { x_m = minimap.PlayerXZ.x, z_m = minimap.PlayerXZ.y, heading_deg = minimap.HeadingDeg };
                byte[] bytes = Encoding.UTF8.GetBytes(JsonUtility.ToJson(body));
                using (var req = new UnityWebRequest(Url("/position"), UnityWebRequest.kHttpVerbPOST))
                {
                    req.uploadHandler = new UploadHandlerRaw(bytes);
                    req.downloadHandler = new DownloadHandlerBuffer();
                    req.SetRequestHeader("Content-Type", "application/json");
                    req.timeout = 5;
                    yield return req.SendWebRequest();
                    if (req.result != UnityWebRequest.Result.Success)
                        Debug.LogWarning($"[Minimap] POST /position failed: {req.responseCode} {req.error}");
                }
            }
            yield return new WaitForSeconds(Mathf.Max(0.2f, positionIntervalSeconds));
        }
    }
}
