using UnityEngine;
using UnityEngine.InputSystem;

[RequireComponent(typeof(CharacterController))]
public class MarsExplorer : MonoBehaviour
{
    [SerializeField] InputActionAsset inputActions;
    [SerializeField] Transform view;
    [SerializeField, Min(0)] float walkSpeed = 2.5f;
    [SerializeField, Min(0)] float fastSpeed = 15f;
    [SerializeField, Min(0)] float mouseSensitivity = 0.08f;
    [SerializeField, Min(0)] float eyeHeight = 1.7f;
    [SerializeField] float gravity = -3.71f;

    CharacterController controller;
    InputActionAsset actions;
    InputAction move, look, sprint;
    float pitch;
    float verticalSpeed;

    void Awake()
    {
        controller = GetComponent<CharacterController>();
        actions = Instantiate(inputActions);
        move = actions.FindAction("Player/Move", true);
        look = actions.FindAction("Player/Look", true);
        sprint = actions.FindAction("Player/Sprint", true);
        view.localPosition = Vector3.up * eyeHeight;
    }

    void OnEnable() => actions.FindActionMap("Player", true).Enable();

    void Start() => SetCursor(true);

    void Update()
    {
        bool recaptured = false;
        if (Keyboard.current?.escapeKey.wasPressedThisFrame == true)
            SetCursor(false);
        else if (Cursor.lockState != CursorLockMode.Locked &&
                 Mouse.current?.leftButton.wasPressedThisFrame == true)
        {
            SetCursor(true);
            recaptured = true;
        }

        var movement = Vector2.zero;
        if (Cursor.lockState == CursorLockMode.Locked)
        {
            if (!recaptured)
            {
                var delta = look.ReadValue<Vector2>() * mouseSensitivity;
                transform.Rotate(Vector3.up, delta.x);
                pitch = Mathf.Clamp(pitch - delta.y, -85, 85);
                view.localRotation = Quaternion.Euler(pitch, 0, 0);
            }
            movement = Vector2.ClampMagnitude(move.ReadValue<Vector2>(), 1);
        }

        float speed = sprint.IsPressed() ? fastSpeed : walkSpeed;
        var velocity = (transform.right * movement.x + transform.forward * movement.y) * speed;
        if (controller.isGrounded && verticalSpeed < 0)
            verticalSpeed = -2;
        verticalSpeed += gravity * Time.deltaTime;
        velocity.y = verticalSpeed;
        controller.Move(velocity * Time.deltaTime);
    }

    void OnApplicationFocus(bool focused)
    {
        if (!focused)
            SetCursor(false);
    }

    void OnDisable()
    {
        actions.Disable();
        SetCursor(false);
    }

    void OnDestroy() => Destroy(actions);

    static void SetCursor(bool captured)
    {
        Cursor.lockState = captured ? CursorLockMode.Locked : CursorLockMode.None;
        Cursor.visible = !captured;
    }
}
