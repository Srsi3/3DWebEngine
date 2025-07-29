use std::collections::HashSet;

use cgmath::{
    Deg, InnerSpace, Matrix4, Point3, Vector3,
    perspective,
};
use winit::keyboard::KeyCode;

pub struct KeyboardInput {
    pressed: HashSet<KeyCode>,
}

impl KeyboardInput {
    pub fn new() -> Self {
        Self { pressed: HashSet::new() }
    }
    pub fn key_press(&mut self, code: KeyCode)   { self.pressed.insert(code); }
    pub fn key_release(&mut self, code: KeyCode) { self.pressed.remove(&code); }
    pub fn is_pressed(&self, code: KeyCode) -> bool { self.pressed.contains(&code) }
}

pub struct Camera {
    pub position: Point3<f32>,
    pub forward:  Vector3<f32>,
    pub right:    Vector3<f32>,
    pub up:       Vector3<f32>,
    pub speed:    f32,   // movement units per second
    pub yaw:      f32,   // radians, left/right
    pub pitch:    f32,   // radians, up/down (clamped)
}

impl Camera {
    pub fn new() -> Self {
        // Start at +Z forward. If you want -Z forward, set forward.z = -1.0 and yaw = PI.
        let position = Point3::new(0.0, 5.0, -10.0);
        let forward  = Vector3::new(0.0, 0.0, 1.0).normalize();
        let up       = Vector3::new(0.0, 1.0, 0.0);
        let right    = forward.cross(up).normalize();

        Self {
            position,
            forward,
            right,
            up,
            speed: 5.0,
            yaw:   0.0,
            pitch: 0.0,
        }
    }

    /// Apply mouse delta (in pixels) to yaw/pitch. Call from WindowEvent::CursorMoved.
    pub fn process_mouse_delta(&mut self, delta_x: f32, delta_y: f32, sensitivity: f32) {
        // Typical: add yaw with +dx, subtract pitch with +dy (so moving mouse up looks up)
        self.yaw   += delta_x * sensitivity;
        self.pitch -= delta_y * sensitivity;
        self.clamp_pitch();
        self.update_axes_from_angles();
    }

    /// Update per-frame: handle rotation keys, then move.
    /// `delta_time` is seconds since last frame.
    pub fn update(&mut self, delta_time: f32, input: &KeyboardInput) {
        // ----- Rotation via keyboard (optional, for testing without mouse) -----
        let rot_speed = 1.5; // radians/sec
        if input.is_pressed(KeyCode::ArrowLeft)  { self.yaw   -= rot_speed * delta_time; }
        if input.is_pressed(KeyCode::ArrowRight) { self.yaw   += rot_speed * delta_time; }
        if input.is_pressed(KeyCode::ArrowUp)    { self.pitch -= rot_speed * delta_time; }
        if input.is_pressed(KeyCode::ArrowDown)  { self.pitch += rot_speed * delta_time; }

        self.clamp_pitch();
        self.update_axes_from_angles();

        // ----- Movement along the rotated axes -----
        let movement = self.speed * delta_time;

        if input.is_pressed(KeyCode::KeyW) { self.position += self.forward * movement; }
        if input.is_pressed(KeyCode::KeyS) { self.position -= self.forward * movement; }
        if input.is_pressed(KeyCode::KeyA) { self.position -= self.right   * movement; }
        if input.is_pressed(KeyCode::KeyD) { self.position += self.right   * movement; }

        // Vertical (noclip) movement
        if input.is_pressed(KeyCode::Space) {
            self.position += self.up * movement;
        }
        if input.is_pressed(KeyCode::ShiftLeft) || input.is_pressed(KeyCode::ShiftRight) {
            self.position -= self.up * movement;
        }
    }

    /// View matrix (right-handed). `look_at_rh` expects `Point3` for eye/center, `Vector3` for up.
    pub fn view_matrix(&self) -> Matrix4<f32> {
        Matrix4::look_at_rh(self.position, self.position + self.forward, self.up)
    }

    /// Basic perspective projection. Pass your swapchain aspect (width/height).
    pub fn projection_matrix(&self, aspect: f32) -> Matrix4<f32> {
        perspective(Deg(60.0), aspect, 0.1, 1_000.0)
    }

    /// Combined view-projection matrix.
    pub fn view_projection(&self, aspect: f32) -> Matrix4<f32> {
        self.projection_matrix(aspect) * self.view_matrix()
    }

    // --- internals ---

    fn clamp_pitch(&mut self) {
        // Prevent gimbal flip; ~±89° is common
        let limit = 89.0_f32.to_radians();
        if self.pitch >  limit { self.pitch =  limit; }
        if self.pitch < -limit { self.pitch = -limit; }
    }

    fn update_axes_from_angles(&mut self) {
        // Forward from yaw/pitch (right-handed, +Z forward at yaw=0).
        // If you want -Z forward at yaw=0, use x =  cos(yaw)*cos(pitch), z = -sin(yaw)*cos(pitch)
        let cy = self.yaw.cos();
        let sy = self.yaw.sin();
        let cp = self.pitch.cos();
        let sp = self.pitch.sin();

        self.forward = Vector3::new(cy * cp, sp, sy * cp).normalize();

        // Derive right and up to keep an orthonormal basis
        self.right = self.forward.cross(Vector3::unit_y()).normalize();
        self.up    = self.right.cross(self.forward).normalize();
    }
}

/// Distance-based culling helper
pub fn should_render(building_pos: Point3<f32>, camera_pos: Point3<f32>, max_distance: f32) -> bool {
    (building_pos - camera_pos).magnitude() < max_distance
}
