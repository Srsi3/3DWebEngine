use cgmath::{Matrix4, Vector3};
use wgpu::util::DeviceExt; // for create_buffer_init
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3], // @location(0)
    pub color:    [f32; 4], // @location(1)
}

impl Vertex {
    pub const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x4, // color
    ];

    pub const fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
    pub index_count:   u32,
}

pub fn create_cube(device: &wgpu::Device) -> Mesh {
    let vertices: Vec<Vertex> = vec![
        // Front (red)
        Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },

        // Back (green)
        Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
    ];

    // Triangle list (two per face)
    let indices: Vec<u16> = vec![
        0, 1, 2,  2, 3, 0, // Front
        4, 5, 6,  6, 7, 4, // Back
        0, 1, 5,  5, 4, 0, // Bottom
        1, 2, 6,  6, 5, 1, // Right
        2, 3, 7,  7, 6, 2, // Top
        3, 0, 4,  4, 7, 3, // Left
    ];
    let index_count = indices.len() as u32;

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    Mesh {
        vertex_buffer,
        index_buffer,
        index_count,
    }
}

/// Builds a grid of cubes; each entry is (Mesh, model_matrix)
pub fn create_city_block(device: &wgpu::Device, grid_size: usize) -> Vec<(Mesh, Matrix4<f32>)> {
    let mut meshes = Vec::new();
    for x in 0..grid_size {
        for z in 0..grid_size {
            let mesh = create_cube(device);
            let translation = Matrix4::from_translation(Vector3::new(
                (x as f32) * 2.0,
                0.0,
                (z as f32) * 2.0,
            ));
            meshes.push((mesh, translation));
        }
    }
    meshes
}
