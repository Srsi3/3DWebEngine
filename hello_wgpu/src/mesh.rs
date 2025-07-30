use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, Vector3};

// ---------- Vertex & Mesh ----------

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub color:    [f32; 4],
}

impl Vertex {
    pub fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 0, offset: 0,  format: wgpu::VertexFormat::Float32x3 },
                wgpu::VertexAttribute { shader_location: 1, offset: 12, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }
}

pub struct Mesh {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer:  wgpu::Buffer,
    pub index_count:   u32,
}

fn upload(device: &wgpu::Device, vertices: &[Vertex], indices: &[u16], label: &str) -> Mesh {
    let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} VB")),
        contents: bytemuck::cast_slice(vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{label} IB")),
        contents: bytemuck::cast_slice(indices),
        usage: wgpu::BufferUsages::INDEX,
    });
    Mesh { vertex_buffer: vb, index_buffer: ib, index_count: indices.len() as u32 }
}

// ---------- Mesh builders ----------

fn build_box_vertices(hx: f32, hy: f32, hz: f32, face_colors: [[f32; 4]; 6]) -> (Vec<Vertex>, Vec<u16>) {
    let positions = [
        // +X
        [ hx,-hy,-hz], [ hx,-hy, hz], [ hx, hy,-hz], [ hx, hy, hz],
        // -X
        [-hx,-hy, hz], [-hx,-hy,-hz], [-hx, hy, hz], [-hx, hy,-hz],
        // +Y
        [-hx, hy,-hz], [ hx, hy,-hz], [-hx, hy, hz], [ hx, hy, hz],
        // -Y
        [-hx,-hy, hz], [ hx,-hy, hz], [-hx,-hy,-hz], [ hx,-hy,-hz],
        // +Z
        [-hx,-hy, hz], [ hx,-hy, hz], [-hx, hy, hz], [ hx, hy, hz],
        // -Z
        [ hx,-hy,-hz], [-hx,-hy,-hz], [ hx, hy,-hz], [-hx, hy,-hz],
    ];

    let mut vertices = Vec::with_capacity(24);
    for face in 0..6 {
        let color = face_colors[face];
        for i in 0..4 {
            vertices.push(Vertex { position: positions[face*4 + i], color });
        }
    }

    let mut indices = Vec::<u16>::with_capacity(6 * 6);
    for f in 0..6 {
        let b = (f * 4) as u16;
        indices.extend_from_slice(&[b, b+1, b+2, b+2, b+1, b+3]);
    }
    (vertices, indices)
}

pub fn create_cuboid(device: &wgpu::Device, w: f32, h: f32, d: f32, color: [f32; 4]) -> Mesh {
    let (v, i) = build_box_vertices(w*0.5, h*0.5, d*0.5, [color; 6]);
    upload(device, &v, &i, "Cuboid")
}

/// 1×1×1 cube centered at origin.
pub fn create_cube(device: &wgpu::Device) -> Mesh {
    let (v, i) = build_box_vertices(
        0.5, 0.5, 0.5,
        [
            [0.9,0.2,0.2,1.0], [0.2,0.9,0.2,1.0], [0.2,0.2,0.9,1.0],
            [0.9,0.9,0.2,1.0], [0.9,0.2,0.9,1.0], [0.2,0.9,0.9,1.0],
        ]
    );
    upload(device, &v, &i, "Cube")
}

/// Wide, low-rise block (warehouse-like).
pub fn create_block_lowrise(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 3.0, 0.8, 2.0, [0.65,0.65,0.70,1.0])
}

/// Tall, slender tower.
pub fn create_tower_highrise(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 0.9, 6.0, 0.9, [0.55,0.60,0.70,1.0])
}

/// Cuboid base + pyramid roof.
pub fn create_pyramid_tower(device: &wgpu::Device) -> Mesh {
    // Base 2.0×1.2×2.0
    let base_w = 2.0; let base_h = 1.2; let base_d = 2.0;
    let base_color = [0.6,0.6,0.65,1.0];

    let (mut vertices, mut indices) = {
        build_box_vertices(base_w*0.5, base_h*0.5, base_d*0.5, [base_color; 6])
    };

    // Roof pyramid
    let roof_h = 0.9;
    let y_base = base_h * 0.5;
    let hx = base_w * 0.5;
    let hz = base_d * 0.5;

    let c0 = Vertex { position: [-hx, y_base, -hz], color: [0.75,0.25,0.25,1.0] };
    let c1 = Vertex { position: [ hx, y_base, -hz], color: [0.75,0.25,0.25,1.0] };
    let c2 = Vertex { position: [-hx, y_base,  hz], color: [0.80,0.30,0.30,1.0] };
    let c3 = Vertex { position: [ hx, y_base,  hz], color: [0.80,0.30,0.30,1.0] };
    let apex = Vertex { position: [0.0, y_base + roof_h, 0.0], color: [0.85,0.35,0.35,1.0] };

    let base_idx = vertices.len() as u16;
    vertices.extend_from_slice(&[c0,c1,c2,c3,apex]);

    let (i0,i1,i2,i3,ia) = (base_idx, base_idx+1, base_idx+2, base_idx+3, base_idx+4);
    indices.extend_from_slice(&[ i0,i1,ia,  i1,i3,ia,  i3,i2,ia,  i2,i0,ia ]);

    upload(device, &vertices, &indices, "Pyramid Tower")
}

/// Vertical quad (1.5×2.5) centered at origin in XY plane, facing +Z.
/// Centered so instance 'pos' places its center correctly for all meshes.
pub fn create_billboard_quad(device: &wgpu::Device) -> Mesh {
    let w = 1.5; let h = 2.5; let hw = w*0.5; let hh = h*0.5;
    let v = vec![
        Vertex { position: [-hw, -hh, 0.0], color: [0.80,0.80,0.85,1.0] },
        Vertex { position: [ hw, -hh, 0.0], color: [0.80,0.80,0.85,1.0] },
        Vertex { position: [-hw,  hh, 0.0], color: [0.85,0.85,0.90,1.0] },
        Vertex { position: [ hw,  hh, 0.0], color: [0.85,0.85,0.90,1.0] },
    ];
    let i: [u16; 6] = [0,1,2, 2,1,3];
    upload(device, &v, &i, "Billboard Quad")
}

pub fn create_ground(device: &wgpu::Device) -> Mesh {
    create_cuboid(device, 2000.0, 0.1, 2000.0, [0.12,0.12,0.14,1.0])
}

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

// ---------- Procedural, chunkable city generation (compact) ----------

#[derive(Copy, Clone, serde::Serialize, serde::Deserialize)]
pub enum BuildingKind { Lowrise, Highrise, Pyramid }

#[derive(Copy, Clone)]
pub struct BuildingRecord {
    pub pos_center: cgmath::Vector3<f32>,
    pub scale:      cgmath::Vector3<f32>,
    pub kind:       BuildingKind,
}

/// Disk form (Serialize) — keep it compact.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct BuildingDisk {
    pub pos:   [f32; 3],
    pub scale: [f32; 3],
    pub kind:  u8,
}

impl From<&BuildingRecord> for BuildingDisk {
    fn from(b: &BuildingRecord) -> Self {
        let kind = match b.kind {
            BuildingKind::Lowrise => 0u8,
            BuildingKind::Highrise => 1u8,
            BuildingKind::Pyramid => 2u8,
        };
        Self {
            pos:   [b.pos_center.x, b.pos_center.y, b.pos_center.z],
            scale: [b.scale.x, b.scale.y, b.scale.z],
            kind,
        }
    }
}
impl From<&BuildingDisk> for BuildingRecord {
    fn from(d: &BuildingDisk) -> Self {
        let kind = match d.kind {
            0 => BuildingKind::Lowrise,
            1 => BuildingKind::Highrise,
            _ => BuildingKind::Pyramid,
        };
        Self {
            pos_center: cgmath::Vector3::new(d.pos[0], d.pos[1], d.pos[2]),
            scale:      cgmath::Vector3::new(d.scale[0], d.scale[1], d.scale[2]),
            kind,
        }
    }
}

#[derive(Copy, Clone)]
pub struct HalfExtents { pub(crate) x: f32, pub(crate) y: f32, pub(crate) z: f32 }
#[derive(Copy, Clone)]
struct PyramidBase { half: HalfExtents, roof_h: f32 }

const BASE_LOW:  HalfExtents = HalfExtents { x: 1.5,  y: 0.4, z: 1.0 };
const BASE_HIGH: HalfExtents = HalfExtents { x: 0.45, y: 3.0, z: 0.45 };
const BASE_PYR:  PyramidBase = PyramidBase { half: HalfExtents { x: 1.0, y: 0.6, z: 1.0 }, roof_h: 0.9 };

/// Public: base half extents per kind (used by culling to compute AABB from scale).
pub fn base_half_for(kind: BuildingKind) -> HalfExtents {
    match kind {
        BuildingKind::Lowrise  => BASE_LOW,
        BuildingKind::Highrise => BASE_HIGH,
        BuildingKind::Pyramid  => HalfExtents {
            x: BASE_PYR.half.x,
            y: (BASE_PYR.half.y * 2.0 + BASE_PYR.roof_h) * 0.5,
            z: BASE_PYR.half.z,
        },
    }
}

#[derive(Clone, Copy)]
pub struct CityGenParams {
    pub lots_x: usize,
    pub lots_z: usize,
    pub lot_w:  f32,
    pub lot_d:  f32,
    pub lot_gap: f32,
    pub road_w_minor: f32,
    pub road_w_major: f32,
    pub major_every:  usize,
    pub blocks_per_chunk_x: usize,
    pub blocks_per_chunk_z: usize,
    pub seed: u64,
}

pub(crate) fn block_world_span(params: &CityGenParams) -> (f32, f32) {
    let span_x = params.lots_x as f32 * (params.lot_w + params.lot_gap) - params.lot_gap;
    let span_z = params.lots_z as f32 * (params.lot_d + params.lot_gap) - params.lot_gap;
    let bx = span_x + params.road_w_minor;
    let bz = span_z + params.road_w_minor;
    (bx, bz)
}

pub fn chunk_world_span(params: &CityGenParams) -> (f32, f32) {
    let (bx, bz) = block_world_span(params);
    let major_x = (params.blocks_per_chunk_x / params.major_every) as f32;
    let major_z = (params.blocks_per_chunk_z / params.major_every) as f32;
    let sx = params.blocks_per_chunk_x as f32 * bx + major_x * (params.road_w_major - params.road_w_minor);
    let sz = params.blocks_per_chunk_z as f32 * bz + major_z * (params.road_w_major - params.road_w_minor);
    (sx, sz)
}

struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x }
    fn unit_f32(&mut self) -> f32 { (self.next() as f64 / u64::MAX as f64) as f32 }
}
fn hash2(a: i32, b: i32) -> u64 {
    let mut x = (a as i64 as i128) as u128 ^ (((b as i64 as i128) << 1) as u128) ^ 0x9E37_79B9_7F4A_7C15u128;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u128);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu128);
    (x ^ (x >> 31)) as u64
}

fn zone_weights(x: f32, z: f32) -> (f32,f32,f32) {
    let dist = x.hypot(z).max(1.0);
    let t = (1.0 - (dist / 1200.0)).clamp(0.0, 1.0);
    let w_high = 0.2 + 0.6 * t;      // downtown bias
    let w_low  = 0.6 - 0.4 * t;
    let w_pyr  = (1.0 - (w_high + w_low)).clamp(0.05, 0.3);
    (w_low.max(0.05), w_high.max(0.05), w_pyr)
}

pub struct CityChunk { pub buildings: Vec<BuildingRecord> }

pub fn generate_city_chunk(params: &CityGenParams, cx: i32, cz: i32) -> CityChunk {
    let (bx, bz) = block_world_span(params);
    let (sx, sz) = chunk_world_span(params);
    let chunk_org_x = cx as f32 * sx;
    let chunk_org_z = cz as f32 * sz;
    let mut rng = XorShift64::new(params.seed ^ hash2(cx, cz));

    let mut buildings = Vec::with_capacity(
        params.blocks_per_chunk_x * params.blocks_per_chunk_z * params.lots_x * params.lots_z
    );

    for bxi in 0..params.blocks_per_chunk_x {
        for bzi in 0..params.blocks_per_chunk_z {
            let major_x = params.major_every > 0 && (bxi % params.major_every == 0);
            let major_z = params.major_every > 0 && (bzi % params.major_every == 0);
            if major_x || major_z { continue; }

            let mut block_x = -0.5*sx + bxi as f32 * bx + params.road_w_minor * 0.5;
            let mut block_z = -0.5*sz + bzi as f32 * bz + params.road_w_minor * 0.5;
            if (bxi % params.major_every) > 0 && ((bxi / params.major_every) > 0) {
                block_x += (params.road_w_major - params.road_w_minor) * ((bxi / params.major_every) as f32);
            }
            if (bzi % params.major_every) > 0 && ((bzi / params.major_every) > 0) {
                block_z += (params.road_w_major - params.road_w_minor) * ((bzi / params.major_every) as f32);
            }

            for lx in 0..params.lots_x {
                for lz in 0..params.lots_z {
                    let x = chunk_org_x + block_x + (lx as f32) * (params.lot_w + params.lot_gap) + params.lot_w * 0.5;
                    let z = chunk_org_z + block_z + (lz as f32) * (params.lot_d + params.lot_gap) + params.lot_d * 0.5;

                    let (w_low, w_high, _w_pyr) = zone_weights(x, z);
                    let pick = rng.unit_f32();
                    let kind = if pick < w_low {
                        BuildingKind::Lowrise
                    } else if pick < (w_low + 0.8) {
                        BuildingKind::Highrise
                    } else {
                        BuildingKind::Pyramid
                    };

                    let sx = 0.85 + 0.30 * rng.unit_f32();
                    let sz = 0.85 + 0.30 * rng.unit_f32();
                    let sy = match kind {
                        BuildingKind::Lowrise  => 0.8 + 0.7 * rng.unit_f32(),
                        BuildingKind::Highrise => {
                            let boost = (1.0 + 1.2 * (1.0 - (x.hypot(z) / 1000.0)).clamp(0.0, 1.0));
                            (0.8 + 1.7 * rng.unit_f32()) * boost
                        }
                        BuildingKind::Pyramid  => 0.8 + 0.8 * rng.unit_f32(),
                    };

                    // center.y is the vertical center (for AABB + rendering)
                    let base_half = base_half_for(kind);
                    let half_y = base_half.y * sy;
                    let center_y = half_y;

                    buildings.push(BuildingRecord {
                        pos_center: Vector3::new(x, center_y, z),
                        scale:      Vector3::new(sx, sy, sz),
                        kind,
                    });
                }
            }
        }
    }

    CityChunk { buildings }
}
