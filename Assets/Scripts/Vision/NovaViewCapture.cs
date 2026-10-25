using System;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

// Capture the headset view without the HUD. An optional close-up shows the target in more detail.
// Read pixels asynchronously and encode the JPEGs off the main thread.
public class NovaViewCapture : MonoBehaviour
{
    public struct Result
    {
        public byte[] jpeg;
        public int width, height;
        public byte[] focusJpeg;   // null without a focus point
        public int focusSize;
        public float focusFov;
        public string error;
    }

    [SerializeField, Min(64)] int width = 1024;
    [SerializeField, Min(64)] int height = 576;
    [Tooltip("Vertical field of view of the image. The headset eye view is wider and per eye.")]
    [SerializeField, Range(30, 100)] float verticalFov = 60;
    [SerializeField, Range(30, 100)] int jpegQuality = 80;
    [Tooltip("Layers left out of the image. The helmet HUD and the NOVA orb are on UI.")]
    [SerializeField] LayerMask excludedLayers = 1 << 5;
    [Tooltip("Square size of the zoomed focus image.")]
    [SerializeField, Min(64)] int focusSize = 768;

    Camera view;

    public bool Busy { get; private set; }

    // Where a world point lands in the last captured image (0-1, origin bottom left; z < 0 = behind the camera).
    public Vector3 ViewportPoint(Vector3 world) => view != null ? view.WorldToViewportPoint(world) : new Vector3(-1, -1, -1);

    // Render now and call done on the main thread when encoding finishes.
    public void Capture(Action<Result> done, Vector3? focus = null, float focusFov = 30)
    {
        if (Busy) { done(new Result { error = "a capture is already running" }); return; }
        var source = Camera.main;
        if (source == null) { done(new Result { error = "no main camera" }); return; }
        int w = width, h = height, f = focusSize;
        var target = RenderTexture.GetTemporary(w, h, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
        RenderTexture zoom = null;
        try
        {
            var camera = Prepare(source, w, h);
            Render(camera, target);
            if (focus.HasValue && (focus.Value - camera.transform.position).sqrMagnitude > 0.01f)
            {
                zoom = RenderTexture.GetTemporary(f, f, 24, RenderTextureFormat.ARGB32, RenderTextureReadWrite.sRGB);
                var fullRotation = camera.transform.rotation;
                camera.transform.rotation = Quaternion.LookRotation(focus.Value - camera.transform.position, Vector3.up);
                camera.fieldOfView = focusFov;
                camera.aspect = 1;
                Render(camera, zoom);
                camera.transform.rotation = fullRotation;   // ViewportPoint keeps describing the full image
                camera.fieldOfView = verticalFov;
                camera.aspect = (float)w / h;
            }
        }
        catch (Exception e)
        {
            RenderTexture.ReleaseTemporary(target);
            if (zoom != null) RenderTexture.ReleaseTemporary(zoom);
            done(new Result { error = $"render failed: {e.Message}" });
            return;
        }
        Busy = true;
        Finish(target, w, h, zoom, f, focusFov, done);
    }

    Camera Prepare(Camera source, int w, int h)
    {
        if (view == null)
        {
            var go = new GameObject("Nova View Capture") { hideFlags = HideFlags.HideAndDontSave };
            view = go.AddComponent<Camera>();
        }
        view.CopyFrom(source);
        view.enabled = false;
        view.stereoTargetEye = StereoTargetEyeMask.None;
        view.ResetWorldToCameraMatrix();
        view.ResetProjectionMatrix();
        view.fieldOfView = verticalFov;
        view.aspect = (float)w / h;
        view.cullingMask = source.cullingMask & ~excludedLayers;
        view.transform.SetPositionAndRotation(source.transform.position, source.transform.rotation);
        var data = view.GetUniversalAdditionalCameraData();
        data.allowXRRendering = false;
        data.renderType = CameraRenderType.Base;
        var sourceData = source.GetUniversalAdditionalCameraData();
        data.renderPostProcessing = sourceData.renderPostProcessing;
        data.antialiasing = sourceData.antialiasing;
        return view;
    }

    static void Render(Camera camera, RenderTexture target)
    {
        var request = new UniversalRenderPipeline.SingleCameraRequest { destination = target };
        if (RenderPipeline.SupportsRenderRequest(camera, request)) RenderPipeline.SubmitRenderRequest(camera, request);
        else
        {
            camera.targetTexture = target;
            camera.Render();
            camera.targetTexture = null;
        }
    }

    async void Finish(RenderTexture target, int w, int h, RenderTexture zoom, int f, float fov, Action<Result> done)
    {
        var result = new Result { width = w, height = h };
        try
        {
            // Copy each readback's pixels right away: a completed request's data does not survive the next await.
            var full = await AsyncGPUReadback.RequestAsync(target, 0, TextureFormat.RGBA32);
            byte[] pixels = full.hasError ? null : full.GetData<byte>().ToArray();
            byte[] zoomPixels = null;
            bool zoomFailed = false;
            if (pixels != null && zoom != null)
            {
                var close = await AsyncGPUReadback.RequestAsync(zoom, 0, TextureFormat.RGBA32);
                zoomFailed = close.hasError;
                if (!zoomFailed) zoomPixels = close.GetData<byte>().ToArray();
            }
            if (pixels == null || zoomFailed) result.error = "GPU readback failed";
            else
            {
                await Awaitable.BackgroundThreadAsync();
                result.jpeg = ImageConversion.EncodeArrayToJPG(pixels, GraphicsFormat.R8G8B8A8_UNorm, (uint)w, (uint)h, 0, jpegQuality);
                if (zoomPixels != null)
                {
                    result.focusJpeg = ImageConversion.EncodeArrayToJPG(zoomPixels, GraphicsFormat.R8G8B8A8_UNorm, (uint)f, (uint)f, 0, jpegQuality);
                    result.focusSize = f;
                    result.focusFov = fov;
                }
                await Awaitable.MainThreadAsync();
            }
        }
        catch (Exception e)
        {
            await Awaitable.MainThreadAsync();
            result.jpeg = result.focusJpeg = null;
            result.error = $"encode failed: {e.Message}";
        }
        RenderTexture.ReleaseTemporary(target);
        if (zoom != null) RenderTexture.ReleaseTemporary(zoom);
        Busy = false;
        done(result);
    }

    void OnDestroy()
    {
        if (view != null) Destroy(view.gameObject);
    }
}
