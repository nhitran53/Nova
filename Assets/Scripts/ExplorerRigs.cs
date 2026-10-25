using UnityEngine;
using UnityEngine.XR.Management;

[DefaultExecutionOrder(-1000)]
public class ExplorerRigs : MonoBehaviour
{
    [SerializeField] GameObject desktopRig;
    [SerializeField] GameObject questRig;

    void Awake()
    {
        bool useXR = Application.platform == RuntimePlatform.Android ||
            XRGeneralSettings.Instance?.Manager?.activeLoader != null;
        desktopRig.SetActive(!useXR);
        questRig.SetActive(useXR);

        if (Application.platform != RuntimePlatform.Android)
            return;

        foreach (var terrain in FindObjectsByType<Terrain>(FindObjectsSortMode.None))
        {
            terrain.heightmapPixelError = terrain.groupingID == 2021 ? 3 : 10;
            terrain.basemapDistance = 500;
            terrain.drawTreesAndFoliage = false;
            terrain.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
        }
    }
}
