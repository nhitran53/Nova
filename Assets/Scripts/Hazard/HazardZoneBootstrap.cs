using UnityEngine;

// No scene edits: survives scene changes and rebinds when the voice client is replaced.
public sealed class HazardZoneBootstrap : MonoBehaviour
{
    HazardZoneOverlay overlay;
    VoiceSessionClient voice;
    float nextLookup;

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
    static void Create()
    {
        if (FindFirstObjectByType<HazardZoneBootstrap>() != null) return;
        var root = new GameObject("Meteor hazard overlay");
        DontDestroyOnLoad(root);
        root.AddComponent<HazardZoneBootstrap>();
    }

    void Awake()
    {
        overlay = gameObject.AddComponent<HazardZoneOverlay>();
    }

    void Update()
    {
        if (Time.unscaledTime < nextLookup) return;
        nextLookup = Time.unscaledTime + 1;
        var found = FindFirstObjectByType<VoiceSessionClient>();
        if (ReferenceEquals(found, voice)) return;
        if (voice != null) voice.OnHazardChanged -= overlay.SetHazard;
        voice = found;
        if (voice != null) voice.OnHazardChanged += overlay.SetHazard;
        overlay.SetHazard(voice != null ? voice.Hazard : null);
    }

    void OnDestroy()
    {
        if (voice != null) voice.OnHazardChanged -= overlay.SetHazard;
    }
}
