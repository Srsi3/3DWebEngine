use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Per-vertex data (matches your shader: location(0)=position, location(1)=color)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position @ location 0
                wgpu::VertexAttribute {
                    shader_location: 0,
                    offset: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                // color @ location 1
                wgpu::VertexAttribute {
                    shader_location: 1,
                    offset: 12,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// A simple mesh wrapper used by your render code.
pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

fn upload(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16], label: &str) -> Mesh {
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} Vertex Buffer")),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} Index Buffer")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Mesh {
        vertex_buffer: vb,
        index_buffer: ib,
        index_count: indices.len() as u32,
    }
}

/// Build a colored box (rectangular prism) centered at the origin with half-sizes (hx,hy,hz).
/// Generates 24 distinct vertices (4 per face) so each face can have its own color.
fn build_box_vertices(hx: f32, hy: f32, hz: f32, face_colors: [[f32; 4]; 6]) -> (Vec<Vertex>, Vec<u16>) {
    // 6 faces: +X, -X, +Y, -Y, +Z, -Z
    // Each face has 4 verts making two triangles (0,1,2, 2,1,3) in face-local order
    let positions = [
        // +X face (right)
        [hx, -hy, -hz], [hx, -hy,  hz], [hx,  hy, -hz], [hx,  hy,  hz],
        // -X face (left)
        [-hx, -hy,  hz], [-hx, -hy, -hz], [-hx,  hy,  hz], [-hx,  hy, -hz],
        // +Y face (top)
        [-hx,  hy, -hz], [ hx,  hy, -hz], [-hx,  hy,  hz], [ hx,  hy,  hz],
        // -Y face (bottom)
        [-hx, -hy,  hz], [ hx, -hy,  hz], [-hx, -hy, -hz], [ hx, -hy, -hz],
        // +Z face (front)
        [-hx, -hy,  hz], [ hx, -hy,  hz], [-hx,  hy,  hz], [ hx,  hy,  hz],
        // -Z face (back)
        [ hx, -hy, -hz], [-hx, -hy, -hz], [ hx,  hy, -hz], [-hx,  hy, -hz],
    ];

    let mut vertices = Vec::with_capacity(24);
    // Assign colors per face (4 vertices each)
    for face in 0..6 {
        let color = face_colors[face];
        for i in 0..4 {
            vertices.push(Vertex {
                position: positions[face * 4 + i],
                color,
            });
        }
    }

    // Two triangles per face = 6 indices per face
    let mut indices: Vec<u16> = Vec::with_capacity(6 * 6);
    for f in 0..6 {
        let base = (f * 4) as u16;
        // (0,1,2) and (2,1,3)
        indices.extend_from_slice(&[
            base, base + 1, base + 2,
            base + 2, base + 1, base + 3,
        ]);
    }

    (vertices, indices)
}

/// Unit cube centered at origin (1x1x1).
pub fn create_cube(device: &wgpu::Device) -> Mesh {
    let hx = 0.5;
    let hy = 0.5;
    let hz = 0.5;

    // Nice distinct face colors (RGBA)
    let face_colors = [
        [0.9, 0.2, 0.2, 1.0], // +X
        [0.2, 0.9, 0.2, 1.0], // -X
        [0.2, 0.2, 0.9, 1.0], // +Y
        [0.9, 0.9, 0.2, 1.0], // -Y
        [0.9, 0.2, 0.9, 1.0], // +Z
        [0.2, 0.9, 0.9, 1.0], // -Z
    ];
    let (v, i) = build_box_vertices(hx, hy, hz, face_colors);
    upload(device, &v, &i, "Cube")
}

/// General rectangular prism (centered), with a single uniform color (applied to all faces).
pub fn create_cuboid(device: &wgpu::Device, w: f32, h: f32, d: f32, color: [f32; 4]) -> Mesh {
    let hx = w * 0.5;
    let hy = h * 0.5;
    let hz = d * 0.5;
    let face_colors = [color; 6];
    let (v, i) = build_box_vertices(hx, hy, hz, face_colors);
    upload(device, &v, &i, "Cuboid")
}

/// Wide, low-rise block (e.g., warehouse)
pub fn create_block_lowrise(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 3.0, 0.8, 2.0, [0.65, 0.65, 0.7, 1.0])
}

/// Tall, slender high-rise tower
pub fn create_tower_highrise(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 0.9, 6.0, 0.9, [0.55, 0.6, 0.7, 1.0])
}

/// Cuboid base + pyramid roof (simple 'house' / 'gable-like' silhouette)
pub fn create_pyramid_tower(device: &wgpu::Device) -> Mesh {
    // Base first (2.0 x 1.2 x 2.0)
    let base_w = 2.0;
    let base_h = 1.2;
    let base_d = 2.0;

    let base_color = [0.6, 0.6, 0.65, 1.0];
    let (mut vertices, mut indices) = {
        let (v, i) = build_box_vertices(base_w * 0.5, base_h * 0.5, base_d * 0.5, [base_color; 6]);
        (v, i)
    };

    // Pyramid roof on top of base (centered, sits on y = +base_h/2)
    // Base of pyramid matches top of cuboid (w x d); apex is higher by `roof_h`.
    let roof_h = 0.9;
    let y_base = base_h * 0.5;
    let hx = base_w * 0.5;
    let hz = base_d * 0.5;

    // Roof vertices: 4 corners of the top rectangle + 1 apex
    let c0 = Vertex { position: [-hx, y_base, -hz], color: [0.7, 0.2, 0.2, 1.0] }; // back-left
    let c1 = Vertex { position: [ hx, y_base, -hz], color: [0.7, 0.2, 0.2, 1.0] }; // back-right
    let c2 = Vertex { position: [-hx, y_base,  hz], color: [0.8, 0.25, 0.25, 1.0] }; // front-left
    let c3 = Vertex { position: [ hx, y_base,  hz], color: [0.8, 0.25, 0.25, 1.0] }; // front-right
    let apex = Vertex { position: [0.0, y_base + roof_h, 0.0], color: [0.85, 0.3, 0.3, 1.0] };

    let base_index = vertices.len() as u16;
    vertices.extend_from_slice(&[c0, c1, c2, c3, apex]);

    // Triangles: four roof faces (c0,c1,apex), (c1,c3,apex), (c3,c2,apex), (c2,c0,apex)
    // Indices relative to base_index
    let (i0, i1, i2, i3, ia) = (base_index, base_index + 1, base_index + 2, base_index + 3, base_index + 4);
    let roof_indices: [u16; 12] = [
        i0, i1, ia,
        i1, i3, ia,
        i3, i2, ia,
        i2, i0, ia,
    ];
    // Optional: you could add two triangles for the "gables" (the rectangle across c0-c1-c3-c2),
    // but the four triangular faces already form a closed pyramid.

    indices.extend_from_slice(&roof_indices);

    upload(device, &vertices, &indices, "Pyramid Tower")
}

/// Vertical quad (width x height), centered at origin, lying in the X-Y plane (Z=0), facing +Z.
/// Good as a very cheap far-LOD surrogate.
/// If you want to make it face the camera later, do it in the vertex shader or rebuild the model matrix.
pub fn create_billboard_quad(device: &wgpu::Device) -> Mesh {
    let w = 1.5;
    let h = 2.5;

    let hw = w * 0.5;
    let c0 = Vertex { position: [-hw, 0.0, 0.0], color: [0.8, 0.8, 0.85, 1.0] }; // bottom-left
    let c1 = Vertex { position: [ hw, 0.0, 0.0], color: [0.8, 0.8, 0.85, 1.0] }; // bottom-right
    let c2 = Vertex { position: [-hw, h,   0.0], color: [0.85, 0.85, 0.9, 1.0] }; // top-left
    let c3 = Vertex { position: [ hw, h,   0.0], color: [0.85, 0.85, 0.9, 1.0] }; // top-right

    let vertices = vec![c0, c1, c2, c3];
    let indices: [u16; 6] = [0, 1, 2, 2, 1, 3];

    upload(device, &vertices, &indices, "Billboard Quad")
}
