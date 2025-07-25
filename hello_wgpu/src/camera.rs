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


//render code for the camera need to reformat for main code block
fn render(&mut self, meshes: &Vec<(Mesh, cgmath::Matrix4<f32>)>, camera: &Camera) {
    let frame = self.surface.get_current_texture().unwrap();
    let view_proj = camera.get_view_projection_matrix();

    let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Render Encoder") });

    for (mesh, translation) in meshes {
        // Create the model-view matrix for each mesh (combining its translation with the camera's view matrix)
        let model_view = view_proj * translation;

        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame.texture.create_view(&wgpu::TextureViewDescriptor::default()),
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 1.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
        });

        rpass.set_pipeline(&self.render_pipeline);
        rpass.set_vertex_buffer(0, &mesh.vertex_buffer, 0, 0);
        rpass.set_index_buffer(&mesh.index_buffer, 0, 0);
        rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
    }

    self.queue.submit(Some(encoder.finish()));
    frame.present();
}

fn should_render(building_position: cgmath::Vector3<f32>, camera_position: cgmath::Vector3<f32>, max_distance: f32) -> bool {
    (building_position - camera_position).magnitude() < max_distance
}
//copilot generated code to render visible buildings based on distance from camera -- not sure if this is the best way to do it
fn render_visible_buildings(&mut self, meshes: &Vec<(Mesh, cgmath::Matrix4<f32>)>, camera: &Camera) {
    let max_distance = 50.0; // Adjust based on your needs
    let camera_position = camera.position;

    for (mesh, translation) in meshes {
        let building_position = translation.w.truncate(); // Extract position from the translation matrix

        if should_render(building_position, camera_position, max_distance) {
            self.render(mesh, translation);
        }
    }
}
