//! Multiplayer-friendly chunk streamer for a finite world.
//!
//! - Tracks multiple viewers (players) and only keeps the union of nearby chunks loaded.
//! - Bakes chunks on first visit via a CityDesigner and persists them.
//! - Compact "placement" data (center + scale + archetype_id).
//! - Supports floating-origin shifts.

use std::collections::{HashMap, HashSet};

use cgmath::Vector3;
use serde::{Serialize, Deserialize};

use crate::assets::{AssetLibrary, BuildingCategory};
use crate::designer_ml::{CityDesigner, DesignContext, Placement};

// ---------- Persistence (v2) ----------

#[derive(Serialize, Deserialize)]
struct PlacementDisk {
    center: [f32; 3],
    scale:  [f32; 3],
    arch:   u16,
}
impl From<&Placement> for PlacementDisk {
    fn from(p: &Placement) -> Self {
        Self {
            center: [p.center.x, p.center.y, p.center.z],
            scale:  [p.scale.x,  p.scale.y,  p.scale.z],
            arch:   p.archetype_id,
        }
    }
}
impl From<&PlacementDisk> for Placement {
    fn from(d: &PlacementDisk) -> Self {
        Self {
            center: Vector3::new(d.center[0], d.center[1], d.center[2]),
            scale:  Vector3::new(d.scale[0],  d.scale[1],  d.scale[2]),
            archetype_id: d.arch,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ChunkFileV2 {
    cx: i32,
    cz: i32,
    placements: Vec<PlacementDisk>,
}

// ---------- World & streaming ----------

#[derive(Clone, Copy)]
pub struct CityGenParamsPublic {
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

fn block_world_span(params: &CityGenParamsPublic) -> (f32, f32) {
    let span_x = params.lots_x as f32 * (params.lot_w + params.lot_gap) - params.lot_gap;
    let span_z = params.lots_z as f32 * (params.lot_d + params.lot_gap) - params.lot_gap;
    let bx = span_x + params.road_w_minor;
    let bz = span_z + params.road_w_minor;
    (bx, bz)
}

pub fn chunk_world_span(params: &CityGenParamsPublic) -> (f32, f32) {
    let (bx, bz) = block_world_span(params);
    let major_x = (params.blocks_per_chunk_x / params.major_every) as f32;
    let major_z = (params.blocks_per_chunk_z / params.major_every) as f32;
    let sx = params.blocks_per_chunk_x as f32 * bx + major_x * (params.road_w_major - params.road_w_minor);
    let sz = params.blocks_per_chunk_z as f32 * bz + major_z * (params.road_w_major - params.road_w_minor);
    (sx, sz)
}

pub type ViewerId = u64;

pub struct ChunkManager {
    /// Public generation params (mirrors mesh::CityGenParams but decoupled).
    pub params: CityGenParamsPublic,

    /// How many chunks around a viewer to keep loaded (Manhattan or square radius).
    pub chunk_radius: i32,

    /// Chunk span in meters.
    pub chunk_span_x: f32,
    pub chunk_span_z: f32,

    /// Loaded chunk -> placements.
    pub loaded: HashMap<(i32,i32), Vec<Placement>>,

    /// Active viewers and their positions.
    viewers: HashMap<ViewerId, (f32, f32)>,

    /// Finite world bounds.
    pub min_cx: i32, pub max_cx: i32,
    pub min_cz: i32, pub max_cz: i32,

    /// If true, newly generated chunks are persisted.
    pub bake_on_miss: bool,

    /// Native: directory path; Web: unused (localStorage).
    pub store_dir: String,
}

impl ChunkManager {
    pub fn new(params_mesh: crate::mesh::CityGenParams,
               chunk_radius: i32,
               bounds: (i32,i32,i32,i32),
               bake_on_miss: bool,
               store_dir: &str) -> Self
    {
        let params = CityGenParamsPublic {
            lots_x: params_mesh.lots_x,
            lots_z: params_mesh.lots_z,
            lot_w: params_mesh.lot_w,
            lot_d: params_mesh.lot_d,
            lot_gap: params_mesh.lot_gap,
            road_w_minor: params_mesh.road_w_minor,
            road_w_major: params_mesh.road_w_major,
            major_every: params_mesh.major_every,
            blocks_per_chunk_x: params_mesh.blocks_per_chunk_x,
            blocks_per_chunk_z: params_mesh.blocks_per_chunk_z,
            seed: params_mesh.seed,
        };
        let (sx, sz) = chunk_world_span(&params);

        Self {
            params,
            chunk_radius,
            chunk_span_x: sx,
            chunk_span_z: sz,
            loaded: HashMap::new(),
            viewers: HashMap::new(),
            min_cx: bounds.0, max_cx: bounds.1, min_cz: bounds.2, max_cz: bounds.3,
            bake_on_miss,
            store_dir: store_dir.to_owned(),
        }
    }

    #[inline]
    pub fn world_to_chunk_coords(&self, x: f32, z: f32) -> (i32, i32) {
        let cx = (x / self.chunk_span_x).floor() as i32;
        let cz = (z / self.chunk_span_z).floor() as i32;
        (cx, cz)
    }

    #[inline]
    fn in_bounds(&self, cx: i32, cz: i32) -> bool {
        cx >= self.min_cx && cx <= self.max_cx && cz >= self.min_cz && cz <= self.max_cz
    }

    /// Register or update a viewer (player) position.
    pub fn set_viewer(&mut self, id: ViewerId, x: f32, z: f32) {
        self.viewers.insert(id, (x, z));
    }

    /// Remove a viewer when they disconnect.
    pub fn remove_viewer(&mut self, id: ViewerId) {
        self.viewers.remove(&id);
        self.prune_unneeded();
    }

    /// Apply floating-origin shift to all loaded chunks.
    pub fn apply_shift(&mut self, offset: Vector3<f32>) {
        for list in self.loaded.values_mut() {
            for p in list {
                p.center -= offset;
            }
        }
        for v in self.viewers.values_mut() {
            v.0 -= offset.x;
            v.1 -= offset.z;
        }
    }

    /// Ensure all viewers have their surrounding chunks loaded. Uses designer to bake on miss.
    pub fn ensure_for_viewers<D: CityDesigner>(
        &mut self,
        designer: &mut D,
        assets: &AssetLibrary,
    ) {
        // Compute union of needed chunks for all viewers.
        let mut want = HashSet::new();
        for (_, (x, z)) in self.viewers.iter() {
            let (ccx, ccz) = self.world_to_chunk_coords(*x, *z);
            for dz in -self.chunk_radius..=self.chunk_radius {
                for dx in -self.chunk_radius..=self.chunk_radius {
                    let cx = ccx + dx;
                    let cz = ccz + dz;
                    if self.in_bounds(cx, cz) {
                        want.insert((cx, cz));
                    }
                }
            }
        }

        // Drop far/out-of-bounds chunks not needed by any viewer.
        self.loaded.retain(|k, _| want.contains(k));

        // Bake or load missing:
        for (cx, cz) in want {
            if self.loaded.contains_key(&(cx, cz)) { continue; }

            // Try V2 store (placements).
            if let Some(cf) = load_chunk_v2(&self.store_dir, cx, cz) {
                let items: Vec<Placement> = cf.placements.iter().map(|d| d.into()).collect();
                self.loaded.insert((cx, cz), items);
                continue;
            }

            // Not in store → design & save.
            let (sx, sz) = chunk_world_span(&self.params);
            let ctx = DesignContext {
                cx, cz,
                world_min: (cx as f32 * sx - 0.5*sx, cz as f32 * sz - 0.5*sz),
                world_max: (cx as f32 * sx + 0.5*sx, cz as f32 * sz + 0.5*sz),
                seed: self.params.seed,
                desired_density: 1.0,
            };

            let items = designer.design_chunk(&ctx, assets);
            if self.bake_on_miss {
                let cf = ChunkFileV2 {
                    cx, cz,
                    placements: items.iter().map(|p| PlacementDisk::from(p)).collect(),
                };
                let _ = save_chunk_v2(&self.store_dir, &cf); // ignore errors
            }
            self.loaded.insert((cx, cz), items);
        }
    }

    /// Remove any loaded chunk that is not within radius of *any* viewer.
    fn prune_unneeded(&mut self) {
        if self.viewers.is_empty() {
            self.loaded.clear();
            return;
        }
        let mut want = HashSet::new();
        for (_, (x, z)) in self.viewers.iter() {
            let (ccx, ccz) = self.world_to_chunk_coords(*x, *z);
            for dz in -self.chunk_radius..=self.chunk_radius {
                for dx in -self.chunk_radius..=self.chunk_radius {
                    let cx = ccx + dx;
                    let cz = ccz + dz;
                    if self.in_bounds(cx, cz) {
                        want.insert((cx, cz));
                    }
                }
            }
        }
        self.loaded.retain(|k, _| want.contains(k));
    }

    // ---- Compatibility shim (if your renderer still expects BuildingRecord/Kind) ----

    /// Map archetype categories back to legacy BuildingKind (for existing code paths).
    pub fn export_legacy_records(&self, assets: &AssetLibrary) -> HashMap<(i32,i32), Vec<crate::mesh::BuildingRecord>> {
        use crate::mesh::BuildingKind;
        let mut out = HashMap::with_capacity(self.loaded.len());
        for (k, list) in self.loaded.iter() {
            let mut v = Vec::with_capacity(list.len());
            for p in list {
                let cat = assets.category_of(p.archetype_id as usize);
                let kind = match cat {
                    BuildingCategory::Lowrise  => BuildingKind::Lowrise,
                    BuildingCategory::Highrise => BuildingKind::Highrise,
                    BuildingCategory::Landmark => BuildingKind::Pyramid,
                };
                v.push(crate::mesh::BuildingRecord {
                    pos_center: p.center,
                    scale: p.scale,
                    kind,
                });
            }
            out.insert(*k, v);
        }
        out
    }
}

// ---------- Storage backends (native fs / web localStorage) ----------

#[cfg(not(target_arch = "wasm32"))]
fn load_chunk_v2(dir: &str, cx: i32, cz: i32) -> Option<ChunkFileV2> {
    let p = std::path::Path::new(dir).join("city_chunks_v2").join(format!("{}_{}.bin", cx, cz));
    std::fs::read(p).ok().and_then(|bytes| bincode::deserialize::<ChunkFileV2>(&bytes).ok())
}
#[cfg(target_arch = "wasm32")]
fn load_chunk_v2(_dir: &str, cx: i32, cz: i32) -> Option<ChunkFileV2> {
    let window = web_sys::window()?;
    let storage = window.local_storage().ok()??;
    let key = format!("city_v2_{}_{}", cx, cz);
    let s = storage.get_item(&key).ok()??;
    let bytes = base64::decode(s).ok()?;
    bincode::deserialize::<ChunkFileV2>(&bytes).ok()
}

#[cfg(not(target_arch = "wasm32"))]
fn save_chunk_v2(dir: &str, cf: &ChunkFileV2) -> std::io::Result<()> {
    let d = std::path::Path::new(dir).join("city_chunks_v2");
    if !d.exists() { std::fs::create_dir_all(&d)?; }
    let p = d.join(format!("{}_{}.bin", cf.cx, cf.cz));
    let bytes = bincode::serialize(cf).expect("bincode serialize");
    std::fs::write(p, bytes)
}
#[cfg(target_arch = "wasm32")]
fn save_chunk_v2(_dir: &str, cf: &ChunkFileV2) -> Result<(), wasm_bindgen::JsValue> {
    let window = web_sys::window().ok_or(wasm_bindgen::JsValue::from_str("no window"))?;
    let storage = window.local_storage()?.ok_or(wasm_bindgen::JsValue::from_str("no localStorage"))?;
    let key = format!("city_v2_{}_{}", cf.cx, cf.cz);
    let bytes = bincode::serialize(cf).map_err(|e| wasm_bindgen::JsValue::from_str(&format!("{e}")))?;
    let s = base64::encode(bytes);
    storage.set_item(&key, &s)
}
