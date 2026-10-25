using UnityEngine;

[ExecuteAlways]
[DisallowMultipleComponent]
public class RoverDamageState : MonoBehaviour
{
    [SerializeField] Transform frontRightWheel;
    [SerializeField] Transform repairedPose;
    [SerializeField] Transform damagedPose;
    [SerializeField] bool damaged = true;

    public bool Damaged => damaged;
    public Transform FrontRightWheel => frontRightWheel;

    void OnEnable() => Apply();
    void OnValidate() => Apply();

    public void SetDamaged(bool value)
    {
        damaged = value;
        Apply();
    }

    [ContextMenu("Demo/Show Damaged")]
    public void ShowDamaged() => SetDamaged(true);

    [ContextMenu("Demo/Show Repaired")]
    public void ShowRepaired() => SetDamaged(false);

    [ContextMenu("Demo/Toggle Damage")]
    public void ToggleDamage() => SetDamaged(!damaged);

    void Apply()
    {
        var pose = damaged ? damagedPose : repairedPose;
        if (frontRightWheel != null && pose != null)
            frontRightWheel.SetPositionAndRotation(pose.position, pose.rotation);
    }
}
