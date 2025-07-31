use std::collections::HashMap;
use cgmath::Vector3;

use crate::designer_ml::{CityDesigner, DesignContext, Placement};
use crate::assets::{AssetLibrary, BuildingCategory};

pub type ViewerId = u32;

/// Runtime placement (what renderer reads)
#[derive(Copy, Clone)]
pub struct RuntimePlacement {
    pub center: Vector3<f32>,
    pub scale:  Vector3<f32>,
    pub archetype_id: u16,
}
#[derive(Clone, Debug)]
pub struct CityGenParams {
    pub lots_x: usize, pub lots_z: usize,
    pub lot_w: f32, pub lot_d: f32, pub lot_gap: f32,
    pub road_w_minor: f32, pub road_w_major: f32, pub major_every: usize,
    pub blocks_per_chunk_x: usize, pub blocks_per_chunk_z: usize,
    pub seed: u64,
}

// world metric sizes
pub fn block_world_span(p: &CityGenParams) -> (f32,f32) {
    let block_w = p.lots_x as f32 * p.lot_w + (p.lots_x-1) as f32 * p.lot_gap + p.road_w_minor;
    let block_d = p.lots_z as f32 * p.lot_d + (p.lots_z-1) as f32 * p.lot_gap + p.road_w_minor;
    (block_w, block_d)
}
pub fn chunk_world_span(p: &CityGenParams) -> (f32,f32) {
    let (bw, bd) = block_world_span(p);
    let mut w = p.blocks_per_chunk_x as f32 * bw;
    let mut d = p.blocks_per_chunk_z as f32 * bd;
    // include major roads
    if p.major_every > 0 {
        w += (p.blocks_per_chunk_x / p.major_every) as f32 * (p.road_w_major - p.road_w_minor);
        d += (p.blocks_per_chunk_z / p.major_every) as f32 * (p.road_w_major - p.road_w_minor);
    }
    (w, d)
}

#[derive(Hash, Eq, PartialEq, Copy, Clone)]
pub(crate) struct ChunkKey(pub i32, pub i32);

fn wrap_coord(c: i32, min_c: i32, max_c: i32) -> i32 {
    let size = max_c - min_c + 1;
    let mut v = (c - min_c) % size;
    if v < 0 { v += size; }
    v + min_c
}

fn wrap_key(cx: i32, cz: i32, bounds: (i32,i32,i32,i32)) -> ChunkKey {
    let (minx, maxx, minz, maxz) = bounds;
    ChunkKey(
        wrap_coord(cx, minx, maxx),
        wrap_coord(cz, minz, maxz),
    )
}

pub struct ChunkManager {
    pub params: CityGenParams,
    pub chunk_radius: i32,
    pub bounds: (i32,i32,i32,i32), // inclusive [minx..maxx]x[minz..maxz]
    pub loaded: HashMap<ChunkKey, Vec<RuntimePlacement>>,
    viewers: HashMap<ViewerId, (f32,f32)>, // x,z in meters

    // baked store path or in-browser storage key prefix
    pub store_prefix: String,

    // torus world span (meters)
    world_span_x: f32,
    world_span_z: f32,
}

impl ChunkManager {
    pub fn new(params: CityGenParams, chunk_radius: i32, bounds: (i32,i32,i32,i32), bake_on_miss: bool, store_prefix: &str) -> Self {
        let (cw, cd) = chunk_world_span(&params);
        Self {
            params,
            chunk_radius: chunk_radius.max(1),
            bounds,
            loaded: HashMap::new(),
            viewers: HashMap::new(),
            store_prefix: store_prefix.to_string(),
            world_span_x: cw * ((bounds.1 - bounds.0 + 1) as f32),
            world_span_z: cd * ((bounds.3 - bounds.2 + 1) as f32),
        }
    }

    #[inline]
    pub fn world_span(&self) -> (f32,f32) { (self.world_span_x, self.world_span_z) }

    pub fn set_viewer(&mut self, id: ViewerId, world_x: f32, world_z: f32) {
        self.viewers.insert(id, (world_x, world_z));
    }

    pub fn apply_shift(&mut self, off: Vector3<f32>) {
        // shift all loaded placements (keep camera-centered continuity)
        for list in self.loaded.values_mut() {
            for p in list.iter_mut() {
                p.center -= off;
            }
        }
        // also shift viewer positions
        for v in self.viewers.values_mut() {
            v.0 -= off.x;
            v.1 -= off.z;
        }
    }

    fn world_to_chunk(&self, x: f32, z: f32) -> (i32, i32) {
        let (cw, cd) = chunk_world_span(&self.params);
        let cx = (x / cw).floor() as i32;
        let cz = (z / cd).floor() as i32;
        (cx, cz)
    }

    fn ensure_chunk(
        &mut self,
        cx: i32, cz: i32,
        designer: &mut dyn CityDesigner,
        assets: &AssetLibrary,
    ) {
        let key = wrap_key(cx, cz, self.bounds);
        if self.loaded.contains_key(&key) { return; }

        // Try load from baked store (omitted for brevity; can add your existing file/localStorage)
        // If not found, design now:
        let ctx = DesignContext { cx: key.0, cz: key.1, seed: self.params.seed };
        let placements = designer.design_chunk(&ctx, assets);

        // Convert to runtime
        let mut rt: Vec<RuntimePlacement> = Vec::with_capacity(placements.len());
        for p in placements {
            rt.push(RuntimePlacement { center: p.center, scale: p.scale, archetype_id: p.archetype_id });
        }

        self.loaded.insert(key, rt);
    }

    pub fn ensure_for_viewers(
        &mut self,
        designer: &mut dyn CityDesigner,
        assets: &AssetLibrary,
    ) {
        for (_vid, (wx, wz)) in self.viewers.clone() {
            let (vcx, vcz) = self.world_to_chunk(wx, wz);
            for dz in -self.chunk_radius..=self.chunk_radius {
                for dx in -self.chunk_radius..=self.chunk_radius {
                    let cx = vcx + dx;
                    let cz = vcz + dz;
                    self.ensure_chunk(cx, cz, designer, assets);
                }
            }
        }
    }

    /// Randomly change a few buildings near viewers (rate: fraction of placements per second).
    pub fn mutate_near(
        &mut self,
        assets: &AssetLibrary,
        rate_per_sec: f32,
        dt: f32,
        radius_chunks: i32,
        seed_add: u64,
    ) {
        if rate_per_sec <= 0.0 { return; }
        let mut want_mut = 0.0;

        let (cw, cd) = chunk_world_span(&self.params);

        for (_id, (wx, wz)) in self.viewers.iter() {
            let (vcx, vcz) = self.world_to_chunk(*wx, *wz);
            for dz in -radius_chunks..=radius_chunks {
                for dx in -radius_chunks..=radius_chunks {
                    let key = wrap_key(vcx + dx, vcz + dz, self.bounds);
                    if let Some(list) = self.loaded.get_mut(&key) {
                        // decide how many to mutate
                        want_mut += (list.len() as f32) * rate_per_sec * dt;
                        while want_mut >= 1.0 && !list.is_empty() {
                            want_mut -= 1.0;
                            // pick random placement and re-roll archetype within same category
                            let idx = (hash2(key.0 ^ key.1, list.len() as i32) ^ seed_add) as usize % list.len();
                            let cat = assets.category_of(list[idx].archetype_id as usize);
                            let ids = assets.indices_by_category(cat);
                            if ids.is_empty() { continue; }
                            // pick different id if possible
                            let mut new_id = ids[(seed_add as usize ^ idx) % ids.len()];
                            if ids.len() > 1 && new_id == list[idx].archetype_id as usize {
                                new_id = ids[(idx + 1) % ids.len()];
                            }
                            list[idx].archetype_id = new_id as u16;

                            // small scale jitter
                            let j = ((seed_add as f32) * 0.000123).sin().abs() * 0.12;
                            list[idx].scale.x = (list[idx].scale.x * (0.95 + j)).clamp(0.7, 1.8);
                            list[idx].scale.y = (list[idx].scale.y * (0.95 + j)).clamp(0.7, 2.5);
                            list[idx].scale.z = (list[idx].scale.z * (0.95 + j)).clamp(0.7, 1.8);

                            // adjust Y to keep on “ground” by base_half
                            let base = assets.base_half(new_id);
                            list[idx].center.y = base.y * list[idx].scale.y;
                        }
                    }
                }
            }
        }
    }
}



// simple hash (same as in designer)
fn hash2(a: i32, b: i32) -> u64 {
    let mut x = (a as i64 as i128) as u128 ^ (((b as i64 as i128) << 1) as u128) ^ 0x9E37_79B9_7F4A_7C15u128;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u128);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu128);
    (x ^ (x >> 31)) as u64
}
