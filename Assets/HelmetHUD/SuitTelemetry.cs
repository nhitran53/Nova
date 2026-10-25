using System;
using UnityEngine;
#if ENABLE_INPUT_SYSTEM
using UnityEngine.InputSystem;
#endif

public enum SuitAlert { Normal, Warning, Critical }

[Serializable]
public struct SuitReadings
{
    public float oxygenPercent;
    public float heartRate;
    public float pressureKpa;
    public float batteryPercent;
    public float radiationMicrosieverts;
    public float temperatureCelsius;
}

public class SuitTelemetry : MonoBehaviour
{
    [Tooltip("Read simulator data through the existing voice client.")]
    [SerializeField] bool useServerTelemetry = true;
    [SerializeField] VoiceSessionClient server;
    [Tooltip("Local demo only when Use Server Telemetry is off.")]
    [SerializeField] bool simulate = true;
    [SerializeField] SuitAlert demoAlert;
    [SerializeField] SuitReadings readings = new SuitReadings
    {
        oxygenPercent = 96, heartRate = 74, pressureKpa = 29.6f,
        batteryPercent = 88, radiationMicrosieverts = 0.05f, temperatureCelsius = 21
    };

    public event Action<SuitReadings> Changed;
    public bool IsSimulated => !useServerTelemetry && simulate;
    public VoiceSessionClient Server => server;
    bool hasServerReading;
    public bool HasData => !useServerTelemetry || (hasServerReading && server != null && server.TelemetryAvailable);
    public SuitReadings Current
    {
        get
        {
            var value = readings;
            if (!useServerTelemetry && demoAlert != SuitAlert.Normal)
                value.radiationMicrosieverts = Mathf.Max(value.radiationMicrosieverts,
                    demoAlert == SuitAlert.Warning ? 0.75f : 2.5f);
            return value;
        }
    }

    float nextUpdate;

    void Update()
    {
#if ENABLE_INPUT_SYSTEM
        if (!useServerTelemetry && Keyboard.current != null && Keyboard.current.f8Key.wasPressedThisFrame)
            CycleDemoAlert();
#endif
        if (Time.unscaledTime < nextUpdate)
            return;
        nextUpdate = Time.unscaledTime + 0.5f;
        if (useServerTelemetry)
        {
            if (server == null) server = FindFirstObjectByType<VoiceSessionClient>();
            if (server != null && server.TelemetryAvailable)
            {
                var value = server.Telemetry;
                readings = new SuitReadings
                {
                    oxygenPercent = value.o2_percent,
                    heartRate = value.heart_rate,
                    pressureKpa = value.suit_pressure_kpa,
                    batteryPercent = value.battery_percent,
                    radiationMicrosieverts = value.radiation_usv_h,
                    temperatureCelsius = value.suit_temp_c
                };
                hasServerReading = true;
            }
            else hasServerReading = false;
        }
        else if (simulate)
        {
            float t = Time.unscaledTime;
            readings.oxygenPercent = 96 + Mathf.Sin(t * 0.04f) * 0.3f;
            readings.heartRate = 74 + Mathf.Sin(t * 0.4f) * 4;
            readings.pressureKpa = 29.6f + Mathf.Sin(t * 0.1f) * 0.1f;
            readings.batteryPercent = 88 + Mathf.Sin(t * 0.02f) * 0.2f;
            readings.radiationMicrosieverts = 0.05f;
            readings.temperatureCelsius = 21 + Mathf.Sin(t * 0.08f) * 0.5f;
        }
        Changed?.Invoke(Current);
    }

    // Call from Unity's main thread when connecting another data source.
    public void SetReadings(SuitReadings value)
    {
        useServerTelemetry = false;
        simulate = false;
        demoAlert = SuitAlert.Normal;
        readings = value;
        Changed?.Invoke(Current);
    }

    [ContextMenu("Demo/Cycle Alert (F8)")]
    public void CycleDemoAlert()
    {
        if (useServerTelemetry) return;
        demoAlert = (SuitAlert)(((int)demoAlert + 1) % 3);
        Changed?.Invoke(Current);
    }

    [ContextMenu("Demo/Clear Alert")]
    public void ClearDemoAlert()
    {
        demoAlert = SuitAlert.Normal;
        Changed?.Invoke(Current);
    }
}
