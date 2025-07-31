use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct CameraUniform {
    pub view_proj: [[f32; 4]; 4],
}

/// Compact instance: world center + non-uniform scale.
/// Rotation is omitted (axis-aligned buildings); add a yaw later if needed.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    pub pos:   [f32; 4], // w unused
    pub scale: [f32; 4], // w unused
    pub misc:  [f32; 4], // x=categoryIdx(0/1/2)  y=archetypeId  z unused
}

pub const fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
    use wgpu::{VertexAttribute, VertexFormat::*};
    wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
        step_mode: wgpu::VertexStepMode::Instance,
        attributes: &[
            VertexAttribute { shader_location: 2, offset: 0,  format: Float32x3 }, // pos.xyz
            VertexAttribute { shader_location: 3, offset: 16, format: Float32x3 }, // scale.xyz
            VertexAttribute { shader_location: 4, offset: 32, format: Float32x3 }, // misc.xyz
        ],
    }
}
