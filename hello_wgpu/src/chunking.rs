use cgmath::InnerSpace;
use std::collections::{HashMap, HashSet};
use cgmath::{Matrix4, Vector3};
use crate::mesh;

/// Keeps chunk streaming + finite world bounds + persistence.
pub struct ChunkManager {
    pub params: mesh::CityGenParams,
    pub chunk_radius: i32,
    pub chunk_span_x: f32,
    pub chunk_span_z: f32,
    pub loaded: HashMap<(i32,i32), Vec<mesh::BuildingRecord>>,

    // finite world bounds (inclusive)
    pub min_cx: i32,
    pub max_cx: i32,
    pub min_cz: i32,
    pub max_cz: i32,

    // persistence
    pub bake_on_miss: bool,
    pub store_dir: String, // native: dir path; web: unused but kept for API symmetry
}

impl ChunkManager {
    pub fn new(params: mesh::CityGenParams, chunk_radius: i32,
               bounds: (i32,i32,i32,i32), bake_on_miss: bool, store_dir: &str) -> Self {
        let (sx, sz) = mesh::chunk_world_span(&params);
        Self {
            params,
            chunk_radius,
            chunk_span_x: sx,
            chunk_span_z: sz,
            loaded: HashMap::new(),
            min_cx: bounds.0, max_cx: bounds.1, min_cz: bounds.2, max_cz: bounds.3,
            bake_on_miss,
            store_dir: store_dir.to_string(),
        }
    }

    pub fn world_to_chunk_coords(&self, x: f32, z: f32) -> (i32, i32) {
        let cx = (x / self.chunk_span_x).floor() as i32;
        let cz = (z / self.chunk_span_z).floor() as i32;
        (cx, cz)
    }

    fn in_bounds(&self, cx: i32, cz: i32) -> bool {
        cx >= self.min_cx && cx <= self.max_cx && cz >= self.min_cz && cz <= self.max_cz
    }

    /// Ensure chunks within radius are present (load → bake → generate).
    /// `origin_shift` is the accumulated floating-origin shift to apply to new items.
    pub fn ensure_for_camera(&mut self, cam_x: f32, cam_z: f32, origin_shift: Vector3<f32>) {
        let (ccx, ccz) = self.world_to_chunk_coords(cam_x, cam_z);

        let mut want = HashSet::new();
        for dz in -self.chunk_radius..=self.chunk_radius {
            for dx in -self.chunk_radius..=self.chunk_radius {
                let cx = ccx + dx;
                let cz = ccz + dz;
                if self.in_bounds(cx, cz) {
                    want.insert((cx, cz));
                }
            }
        }

        // Drop far / out of bounds
        self.loaded.retain(|k, _| want.contains(k));

        for (cx, cz) in want {
            if self.loaded.contains_key(&(cx, cz)) { continue; }

            // 1) Try load from store
            let mut loaded_from_store = None;
            #[cfg(not(target_arch = "wasm32"))]
            {
                if let Some(cf) = crate::city_store::native::load_chunk(&self.store_dir, cx, cz) {
                    loaded_from_store = Some(cf);
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(cf) = crate::city_store::web::load_chunk(&self.store_dir, cx, cz) {
                    loaded_from_store = Some(cf);
                }
            }

            if let Some(cf) = loaded_from_store {
                let mut items: Vec<mesh::BuildingRecord> = cf.buildings.iter().map(|d| d.into()).collect();
                if origin_shift.magnitude2() > 0.0 {
                    for b in &mut items { b.pos_center -= origin_shift; }
                }
                self.loaded.insert((cx, cz), items);
                continue;
            }

            // 2) Not in store → generate
            let chunk = mesh::generate_city_chunk(&self.params, cx, cz);
            let mut items = chunk.buildings;

            // apply current origin shift
            if origin_shift.magnitude2() > 0.0 {
                for b in &mut items { b.pos_center -= origin_shift; }
            }

            // 3) Bake on miss (persist)
            if self.bake_on_miss {
                let cf = crate::city_store::ChunkFile {
                    cx, cz,
                    buildings: items.iter().map(|b| mesh::BuildingDisk::from(b)).collect(),
                };
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let _ = crate::city_store::native::save_chunk(&self.store_dir, &cf);
                }
                #[cfg(target_arch = "wasm32")]
                {
                    let _ = crate::city_store::web::save_chunk(&self.store_dir, &cf);
                }
            }

            self.loaded.insert((cx, cz), items);
        }
    }

    /// Apply a floating-origin shift to all loaded chunks.
    pub fn apply_shift(&mut self, offset: Vector3<f32>) {
        for list in self.loaded.values_mut() {
            for b in list {
                b.pos_center -= offset;
            }
        }
    }
}
