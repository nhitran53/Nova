using TMPro;
using UnityEngine;
using UnityEngine.UI;

// Route guidance in the top-right corner.
public class NavigationHud : MonoBehaviour
{
    [Tooltip("Leave empty to use RouteNavigator.Current.")]
    [SerializeField] RouteNavigator navigator;
    [Tooltip("Center of the readout in Helmet HUD canvas units.")]
    [SerializeField] Vector2 hudSlot = new Vector2(440, 245);
    [SerializeField] Vector2 screenMargin = new Vector2(40, 40);

    static readonly Color Cyan = new Color(0.35f, 0.9f, 1);
    static readonly Color Amber = new Color(1, 0.75f, 0.3f);
    static readonly Color Green = new Color(0.45f, 1f, 0.5f);

    Transform attachedTo;
    TMP_Text text;
    GameObject screenCanvas;
    float nextRefresh;
    string shown;

    void LateUpdate()
    {
        var cam = Camera.main;
        if (cam == null) return;
        if (attachedTo != cam.transform) Attach(cam);
        if (Time.unscaledTime < nextRefresh) return;
        nextRefresh = Time.unscaledTime + 0.2f;

        var nav = navigator != null ? navigator : RouteNavigator.Current;
        var route = nav != null ? nav.ActiveRoute : null;
        string value = route == null ? "" : Format(nav, route, cam.transform);
        if (value == shown) return;
        shown = value;
        text.text = value;
        text.color = nav == null ? Cyan : nav.Status == NavigationStatus.OffRoute ? Amber :
            nav.Status == NavigationStatus.Arrived ? Green : Cyan;
    }

    static string Format(RouteNavigator nav, NavigationRoute route, Transform head)
    {
        string name = route.destinationName.ToUpperInvariant();
        if (nav.Status == NavigationStatus.Arrived)
            return $"<size=70%>DESTINATION</size>\n<b>{name}</b>\nARRIVED";
        int eta = Mathf.CeilToInt(nav.EtaSeconds);
        string status = nav.Status == NavigationStatus.OffRoute
            ? $"OFF ROUTE  {Mathf.RoundToInt(nav.DistanceFromTrail)} m"
            : "ON ROUTE";
        return $"<size=70%>DESTINATION</size>\n<b>{name}</b>  {Mathf.RoundToInt(nav.RemainingMeters)} m\n" +
            $"ETA {eta / 60}:{eta % 60:00}   {Direction(head, nav.NextWaypoint)}\n<size=80%>{status}</size>";
    }

    static string Direction(Transform head, Vector3 target)
    {
        Vector3 forward = Vector3.ProjectOnPlane(head.forward, Vector3.up);
        Vector3 to = Vector3.ProjectOnPlane(target - head.position, Vector3.up);
        if (forward.sqrMagnitude < 1e-4f || to.sqrMagnitude < 0.25f) return "";
        int angle = Mathf.RoundToInt(Vector3.SignedAngle(forward, to, Vector3.up));
        if (Mathf.Abs(angle) <= 10) return "NEXT AHEAD";
        if (Mathf.Abs(angle) >= 160) return "NEXT BEHIND";
        return angle < 0 ? $"NEXT {-angle}° LEFT" : $"NEXT {angle}° RIGHT";
    }

    void Attach(Camera cam)
    {
        attachedTo = cam.transform;
        if (text != null) Destroy(text.gameObject);
        shown = null;
        var hud = cam.GetComponentInChildren<HelmetHUD>(true);
        if (hud != null)
        {
            if (screenCanvas != null) screenCanvas.SetActive(false);
            text = HudText.Create(hud, "Navigation", hudSlot, new Vector2(300, 140), 24);
            text.alignment = TextAlignmentOptions.TopRight;
            return;
        }
        if (screenCanvas == null)
        {
            screenCanvas = new GameObject("Navigation Canvas", typeof(RectTransform), typeof(Canvas), typeof(CanvasScaler));
            screenCanvas.transform.SetParent(transform, false);
            var canvas = screenCanvas.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 10;
            var scaler = screenCanvas.GetComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1920, 1080);
            scaler.matchWidthOrHeight = 1;
        }
        screenCanvas.SetActive(true);
        var rt = new GameObject("Navigation", typeof(RectTransform)).GetComponent<RectTransform>();
        rt.SetParent(screenCanvas.transform, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = Vector2.one;
        rt.anchoredPosition = -screenMargin;
        rt.sizeDelta = new Vector2(420, 200);
        text = rt.gameObject.AddComponent<TextMeshProUGUI>();
        text.fontSize = 30;
        text.alignment = TextAlignmentOptions.TopRight;
        text.raycastTarget = false;
        text.outlineWidth = 0.2f;
    }
}

public static class HudText
{
    public static TMP_Text Create(HelmetHUD hud, string name, Vector2 position, Vector2 size, float fontSize)
    {
        var style = hud.GetComponentInChildren<TMP_Text>(true);
        var rt = new GameObject(name, typeof(RectTransform)).GetComponent<RectTransform>();
        rt.gameObject.layer = hud.gameObject.layer;
        rt.SetParent(hud.transform, false);
        rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(0.5f, 0.5f);
        rt.anchoredPosition = position;
        rt.sizeDelta = size;
        var text = rt.gameObject.AddComponent<TextMeshProUGUI>();
        if (style != null)
        {
            text.font = style.font;
            text.fontSharedMaterial = style.fontSharedMaterial;
        }
        text.fontSize = fontSize;
        text.alignment = TextAlignmentOptions.Center;
        text.raycastTarget = false;
        text.textWrappingMode = TextWrappingModes.NoWrap;
        return text;
    }
}
