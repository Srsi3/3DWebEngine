use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

/// Compact instance: world center + non-uniform scale.
/// Rotation is omitted (axis-aligned buildings); add a yaw later if needed.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct InstanceRaw {
    pub pos:   [f32; 4], // pos.xyz, w unused
    pub scale: [f32; 4], // scale.xyz, w unused
}

pub fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            wgpu::VertexAttribute { shader_location: 2, offset: 0,  format: wgpu::VertexFormat::Float32x4 },
            wgpu::VertexAttribute { shader_location: 3, offset: 16, format: wgpu::VertexFormat::Float32x4 },
        ],
    }
}
