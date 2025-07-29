use cgmath::InnerSpace;
use std::collections::{HashMap, HashSet};
use cgmath::{Matrix4, Vector3};
use crate::mesh;

/// Keeps chunk streaming state and calls `mesh::generate_city_chunk`.
pub struct ChunkManager {
    pub params: mesh::CityGenParams,
    pub chunk_radius: i32,
    pub chunk_span_x: f32,
    pub chunk_span_z: f32,
    pub loaded: HashMap<(i32,i32), Vec<mesh::GeneratedBuilding>>,
}

impl ChunkManager {
    pub fn new(params: mesh::CityGenParams, chunk_radius: i32) -> Self {
        let (sx, sz) = mesh::chunk_world_span(&params);
        Self {
            params,
            chunk_radius,
            chunk_span_x: sx,
            chunk_span_z: sz,
            loaded: HashMap::new(),
        }
    }

    pub fn world_to_chunk_coords(&self, x: f32, z: f32) -> (i32, i32) {
        let cx = (x / self.chunk_span_x).floor() as i32;
        let cz = (z / self.chunk_span_z).floor() as i32;
        (cx, cz)
    }

    /// Ensure chunks in a (2r+1)^2 square around camera are loaded.
    /// `origin_shift` is the accumulated floating-origin shift (to place newly generated chunks correctly).
    pub fn ensure_for_camera(&mut self, cam_x: f32, cam_z: f32, origin_shift: Vector3<f32>) {
        let (ccx, ccz) = self.world_to_chunk_coords(cam_x, cam_z);

        let mut want = HashSet::new();
        for dz in -self.chunk_radius..=self.chunk_radius {
            for dx in -self.chunk_radius..=self.chunk_radius {
                want.insert((ccx + dx, ccz + dz));
            }
        }

        // drop far
        self.loaded.retain(|k, _| want.contains(k));

        // create missing
        for (cx, cz) in want {
            if !self.loaded.contains_key(&(cx, cz)) {
                let chunk = mesh::generate_city_chunk(&self.params, cx, cz);
                let mut items = chunk.buildings;
                if origin_shift.magnitude() > 0.0 {
                    let t = Matrix4::<f32>::from_translation(-origin_shift);
                    for b in &mut items {
                        b.model_near_mid = t * b.model_near_mid;
                        b.model_far      = t * b.model_far;
                        b.center        -= origin_shift;
                    }
                }
                self.loaded.insert((cx, cz), items);
            }
        }
    }

    /// Apply a floating-origin shift to all loaded chunks.
    pub fn apply_shift(&mut self, offset: Vector3<f32>) {
        let t = Matrix4::<f32>::from_translation(-offset);
        for list in self.loaded.values_mut() {
            for b in list {
                b.model_near_mid = t * b.model_near_mid;
                b.model_far      = t * b.model_far;
                b.center        -= offset;
            }
        }
    }
}
