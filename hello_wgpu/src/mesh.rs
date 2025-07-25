#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Vertex {
    position: [f32; 3], // x, y, z
    color: [f32; 4],    // r, g, b, a
}

impl Vertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![0 => Float3, 1 => Float4];
}

struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u16>,  // Indices for indexed drawing (triangle list)
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
}

pub fn create_cube(device: &wgpu::Device) -> Mesh {
    let vertices = vec![
        // Front face
        Vertex { position: [-0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [ 0.5, -0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [ 0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },
        Vertex { position: [-0.5,  0.5,  0.5], color: [1.0, 0.0, 0.0, 1.0] },

        // Back face
        Vertex { position: [-0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [ 0.5, -0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [ 0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
        Vertex { position: [-0.5,  0.5, -0.5], color: [0.0, 1.0, 0.0, 1.0] },
    ];

    let indices: Vec<u16> = vec![
        0, 1, 2, 2, 3, 0, // Front
        4, 5, 6, 6, 7, 4, // Back
        0, 1, 5, 5, 4, 0, // Bottom
        1, 2, 6, 6, 5, 1, // Right
        2, 3, 7, 7, 6, 2, // Top
        3, 0, 4, 4, 7, 3, // Left
    ];

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
        vertices,
        indices,
        vertex_buffer,
        index_buffer,
    }
}


pub fn create_city_block(device: &wgpu::Device, grid_size: usize) -> Vec<Mesh> {
    let mut meshes = Vec::new();

    for x in 0..grid_size {
        for z in 0..grid_size {
            let mesh = create_cube(device);
            let translation = cgmath::Matrix4::from_translation(cgmath::Vector3::new(x as f32 * 2.0, 0.0, z as f32 * 2.0));
            meshes.push((mesh, translation));
        }
    }

    meshes
}
pub fn create_city_block(device: &wgpu::Device, grid_size: usize) -> Vec<Mesh> {
    let mut meshes = Vec::new();

    for x in 0..grid_size {
        for z in 0..grid_size {
            let mesh = create_cube(device);
            let translation = cgmath::Matrix4::from_translation(cgmath::Vector3::new(x as f32 * 2.0, 0.0, z as f32 * 2.0));
            meshes.push((mesh, translation));
        }
    }

    meshes
}
