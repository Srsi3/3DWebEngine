struct Camera {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
    view_matrix: cgmath::Matrix4<f32>,
    projection_matrix: cgmath::Matrix4<f32>,
}

impl Camera {
    fn new() -> Self {
        let position = cgmath::Vector3::new(0.0, 5.0, -10.0); // Start behind the city
        let rotation = cgmath::Quaternion::from_angle_x(cgmath::Deg(0.0));
        let view_matrix = cgmath::Matrix4::look_at_rh(position, cgmath::Vector3::zero(), cgmath::Vector3::unit_y());
        let projection_matrix = cgmath::perspective(cgmath::Deg(45.0), 16.0/9.0, 0.1, 100.0);

        Camera {
            position,
            rotation,
            view_matrix,
            projection_matrix,
        }
    }

    fn update(&mut self, delta_time: f32) {
        // Move the camera with keyboard inputs (W, A, S, D)
        // Update `position` and `view_matrix` based on input
    }

    fn get_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        self.projection_matrix * self.view_matrix
    }
}

