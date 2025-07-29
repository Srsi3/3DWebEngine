use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, Vector3};

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
    for face in 0..6 {
        let color = face_colors[face];
        for i in 0..4 {
            vertices.push(Vertex {
                position: positions[face * 4 + i],
                color,
            });
        }
    }

    let mut indices: Vec<u16> = Vec::with_capacity(6 * 6);
    for f in 0..6 {
        let base = (f * 4) as u16;
        indices.extend_from_slice(&[
            base, base + 1, base + 2,
            base + 2, base + 1, base + 3,
        ]);
    }

    (vertices, indices)
}

// ----------------- BASIC MESH BUILDERS -----------------

/// Unit cube centered at origin (1x1x1).
pub fn create_cube(device: &wgpu::Device) -> Mesh {
    let hx = 0.5;
    let hy = 0.5;
    let hz = 0.5;

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

/// Cuboid base + pyramid roof (simple 'roofed' tower)
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

    // Pyramid roof on top of base
    let roof_h = 0.9;
    let y_base = base_h * 0.5;
    let hx = base_w * 0.5;
    let hz = base_d * 0.5;

    let c0 = Vertex { position: [-hx, y_base, -hz], color: [0.7, 0.2, 0.2, 1.0] };
    let c1 = Vertex { position: [ hx, y_base, -hz], color: [0.7, 0.2, 0.2, 1.0] };
    let c2 = Vertex { position: [-hx, y_base,  hz], color: [0.8, 0.25, 0.25, 1.0] };
    let c3 = Vertex { position: [ hx, y_base,  hz], color: [0.8, 0.25, 0.25, 1.0] };
    let apex = Vertex { position: [0.0, y_base + roof_h, 0.0], color: [0.85, 0.3, 0.3, 1.0] };

    let base_index = vertices.len() as u16;
    vertices.extend_from_slice(&[c0, c1, c2, c3, apex]);

    // Triangles: four roof faces
    let (i0, i1, i2, i3, ia) = (base_index, base_index + 1, base_index + 2, base_index + 3, base_index + 4);
    let roof_indices: [u16; 12] = [
        i0, i1, ia,
        i1, i3, ia,
        i3, i2, ia,
        i2, i0, ia,
    ];
    indices.extend_from_slice(&roof_indices);

    upload(device, &vertices, &indices, "Pyramid Tower")
}

/// Vertical quad (width x height), centered at origin, lying in the X-Y plane (Z=0), facing +Z.
/// Good as a very cheap far-LOD surrogate.
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

// Ground (large thin rectangle using the same "cuboid" builder)
pub fn create_ground(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 2000.0, 0.1, 2000.0, [0.12, 0.12, 0.14, 1.0])
}

// ----------------- CITY GENERATION API -----------------

/// Mesh group for the city (created once with the device)
pub struct CityMeshes {
    pub lowrise:   Mesh,
    pub highrise:  Mesh,
    pub pyramid:   Mesh,
    pub billboard: Mesh,
    pub ground:    Mesh,
}

pub fn create_city_meshes(device: &wgpu::Device) -> CityMeshes {
    CityMeshes {
        lowrise:   create_block_lowrise(device),
        highrise:  create_tower_highrise(device),
        pyramid:   create_pyramid_tower(device),
        billboard: create_billboard_quad(device),
        ground:    create_ground(device),
    }
}

/// Building types; used by hello_wgpu to route to the right mesh
#[derive(Copy, Clone)]
pub enum BuildingKind {
    Lowrise,
    Highrise,
    Pyramid,
}

/// CPU-side description of a single generated building.
pub struct GeneratedBuilding {
    pub model_near_mid: Matrix4<f32>,
    pub model_far:      Matrix4<f32>,
    pub center:         Vector3<f32>,
    pub half:           Vector3<f32>,
    pub kind:           BuildingKind,
}

/// Basic parameters for the city layout
pub struct CityParams {
    pub blocks_x: usize,
    pub blocks_z: usize,
    pub road_w:  f32,
    pub lot_w:   f32,
    pub lot_d:   f32,
    pub lots_x:  usize,
    pub lots_z:  usize,
    pub lot_gap: f32,
    pub seed:    u32,
}

// Unscaled half-extents for a box-like mesh
#[derive(Copy, Clone)]
struct HalfExtents { x: f32, y: f32, z: f32 }

// Pyramid base data: base half extents + roof height (unscaled)
#[derive(Copy, Clone)]
struct PyramidBase { half: HalfExtents, roof_h: f32 }

// Deterministic, super lightweight PRNG
struct XorShift32(u32);
impl XorShift32 {
    fn new(seed: u32) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        self.0 = x; x
    }
    fn unit_f32(&mut self) -> f32 {
        (self.next() as f32) / (u32::MAX as f32)
    }
}

/// Generate a procedural city (buildings only) and a ground model matrix.
/// - Returns: (buildings, ground_model_matrix)
pub fn generate_city_instances(params: &CityParams) -> (Vec<GeneratedBuilding>, Matrix4<f32>) {
    // Base (unscaled) AABBs for kinds (from mesh definitions)
    let base_low   = HalfExtents { x: 1.5,  y: 0.4, z: 1.0 };   // 3.0 x 0.8 x 2.0
    let base_high  = HalfExtents { x: 0.45, y: 3.0, z: 0.45 };  // 0.9 x 6.0 x 0.9
    // Pyramid tower: base (2.0 x 1.2 x 2.0) + roof 0.9
    let base_pyr   = PyramidBase { half: HalfExtents { x: 1.0, y: 0.6, z: 1.0 }, roof_h: 0.9 };

    // Simple deterministic RNG
    let mut rng = XorShift32::new(0x1234_5678 ^ params.seed);

    // Lot and block spans
    let lot_span_x = params.lots_x as f32 * (params.lot_w + params.lot_gap) - params.lot_gap;
    let lot_span_z = params.lots_z as f32 * (params.lot_d + params.lot_gap) - params.lot_gap;
    let block_span_x = lot_span_x + params.road_w;
    let block_span_z = lot_span_z + params.road_w;

    let mut buildings: Vec<GeneratedBuilding> =
        Vec::with_capacity(params.blocks_x * params.blocks_z * params.lots_x * params.lots_z);

    for bx in 0..params.blocks_x {
        for bz in 0..params.blocks_z {
            // Block origin (SW corner of its lot grid), centered around (0,0)
            let block_origin_x = bx as f32 * block_span_x - (params.blocks_x as f32 * block_span_x) * 0.5;
            let block_origin_z = bz as f32 * block_span_z - (params.blocks_z as f32 * block_span_z) * 0.5;

            for lx in 0..params.lots_x {
                for lz in 0..params.lots_z {
                    let x = block_origin_x + (lx as f32) * (params.lot_w + params.lot_gap) + params.lot_w * 0.5;
                    let z = block_origin_z + (lz as f32) * (params.lot_d + params.lot_gap) + params.lot_d * 0.5;

                    // Weighted kind choice
                    let r = (rng.next() % 100) as u32;
                    let kind = if r < 45 { BuildingKind::Lowrise }
                               else if r < 85 { BuildingKind::Highrise }
                               else { BuildingKind::Pyramid };

                    // Subtle variation per dimension
                    let sx = 0.85 + 0.30 * (rng.unit_f32());
                    let sz = 0.85 + 0.30 * (rng.unit_f32());
                    let sy = match kind {
                        BuildingKind::Lowrise  => 0.8 + 0.7 * rng.unit_f32(),  // 0.8..1.5
                        BuildingKind::Highrise => 0.8 + 1.7 * rng.unit_f32(),  // 0.8..2.5
                        BuildingKind::Pyramid  => 0.8 + 0.8 * rng.unit_f32(),  // 0.8..1.6
                    };

                    // Compute models + AABB (object sits on ground at y=0)
                    let (model_near_mid, center, half, model_far) =
                        build_models_and_aabb(kind, sx, sy, sz, x, z, base_low, base_high, base_pyr);

                    buildings.push(GeneratedBuilding {
                        model_near_mid,
                        model_far,
                        center,
                        half,
                        kind,
                    });
                }
            }
        }
    }

    // Ground: large, flat near y=0; place it slightly below to avoid z-fighting
    let ground_model = Matrix4::<f32>::from_translation(Vector3::new(0.0, -0.05, 0.0));

    (buildings, ground_model)
}

fn build_models_and_aabb(
    kind: BuildingKind,
    sx: f32, sy: f32, sz: f32,
    x: f32, z: f32,
    base_low: HalfExtents,
    base_high: HalfExtents,
    base_pyr: PyramidBase,
) -> (Matrix4<f32>, Vector3<f32>, Vector3<f32>, Matrix4<f32>) {

    // Model for near/mid + culling center/half
    let (model_near_mid, center, half) = match kind {
        BuildingKind::Lowrise => {
            let half_scaled = Vector3::new(base_low.x * sx, base_low.y * sy, base_low.z * sz);
            let t = Matrix4::<f32>::from_translation(Vector3::new(x, half_scaled.y, z));
            let s = Matrix4::<f32>::from_nonuniform_scale(sx, sy, sz);
            let model = t * s;
            let center = Vector3::new(x, half_scaled.y, z);
            (model, center, half_scaled)
        }
        BuildingKind::Highrise => {
            let half_scaled = Vector3::new(base_high.x * sx, base_high.y * sy, base_high.z * sz);
            let t = Matrix4::<f32>::from_translation(Vector3::new(x, half_scaled.y, z));
            let s = Matrix4::<f32>::from_nonuniform_scale(sx, sy, sz);
            let model = t * s;
            let center = Vector3::new(x, half_scaled.y, z);
            (model, center, half_scaled)
        }
        BuildingKind::Pyramid => {
            // Base is centered with half.y = 0.6; roof apex adds +0.9 above +base_h/2
            let base_h = base_pyr.half.y * 2.0;
            let total_h_unscaled = base_h + base_pyr.roof_h;
            let t_y = (base_h * 0.5) * sy;
            let t = Matrix4::<f32>::from_translation(Vector3::new(x, t_y, z));
            let s = Matrix4::<f32>::from_nonuniform_scale(sx, sy, sz);
            let model = t * s;
            let center_y = (total_h_unscaled * 0.5) * sy;
            let center = Vector3::new(x, center_y, z);
            let half_scaled = Vector3::new(base_pyr.half.x * sx, total_h_unscaled * 0.5 * sy, base_pyr.half.z * sz);
            (model, center, half_scaled)
        }
    };

    // Billboard model (vertical quad base is 1.5x2.5 in create_billboard_quad)
    let bill_w = (half.x * 2.0).max(0.5);
    let bill_h = (half.y * 2.0).max(0.5);
    let sx_bill = bill_w / 1.5;
    let sy_bill = bill_h / 2.5;
    let t_bill = Matrix4::<f32>::from_translation(Vector3::new(x, bill_h * 0.5, z));
    let s_bill = Matrix4::<f32>::from_nonuniform_scale(sx_bill, sy_bill, 1.0);
    let model_far = t_bill * s_bill;

    (model_near_mid, center, half, model_far)
}

