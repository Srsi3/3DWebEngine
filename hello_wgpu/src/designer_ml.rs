//! Pluggable city designer: rule-based (tiny) and an optional ML-enhanced designer.
//! The ML path is *offline-or-on-bake*, not per-frame, to keep the runtime lightweight.

use cgmath::Vector3;
use crate::assets::{AssetLibrary, BuildingCategory};

/// What the designer outputs per lot (runtime only; no serde needed).
#[derive(Copy, Clone)]
pub struct Placement {
    pub center: Vector3<f32>,
    pub scale:  Vector3<f32>,
    pub archetype_id: u16, // index into AssetLibrary
}

/// Chunk request context (bounds, seed, etc.) – runtime only.
pub struct DesignContext {
    pub cx: i32,
    pub cz: i32,
    pub world_min: (f32, f32),
    pub world_max: (f32, f32),
    pub seed: u64,
    pub desired_density: f32, // 0..1
}

/// Trait for any city designer (rule-based, ML-guided, server-fed, etc.)
pub trait CityDesigner {
    fn design_chunk(&mut self, ctx: &DesignContext, assets: &AssetLibrary) -> Vec<Placement>;
}

// -------------- Tiny RNG --------------

struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x }
    fn unit_f32(&mut self) -> f32 { (self.next() as f64 / u64::MAX as f64) as f32 }
}

// -------------- Rule-based designer (deterministic, tiny) --------------

pub struct RuleDesigner {
    pub params: crate::mesh::CityGenParams,
}

impl RuleDesigner {
    fn zone_weights(&self, x: f32, z: f32) -> (f32,f32,f32) {
        let dist = x.hypot(z).max(1.0);
        let t = (1.0 - (dist / 1200.0)).clamp(0.0, 1.0);
        let w_high = 0.2 + 0.6 * t;
        let w_low  = 0.6 - 0.4 * t;
        let w_land = (1.0 - (w_high + w_low)).clamp(0.05, 0.3);
        (w_low.max(0.05), w_high.max(0.05), w_land)
    }

    /// Pick an archetype index from a category, using RNG to vary variants if available.
    fn pick_archetype(assets: &AssetLibrary, cat: BuildingCategory, rng: &mut XorShift64) -> Option<usize> {
        let ids = assets.indices_by_category(cat);
        if ids.is_empty() { return None; }
        let k = (rng.next() as usize) % ids.len();
        Some(ids[k])
    }
}

impl CityDesigner for RuleDesigner {
    fn design_chunk(&mut self, ctx: &DesignContext, assets: &AssetLibrary) -> Vec<Placement> {
        use crate::mesh::{chunk_world_span, block_world_span};
        let (bx, bz) = block_world_span(&self.params);
        let (sx, sz) = chunk_world_span(&self.params);
        let chunk_org_x = ctx.cx as f32 * sx;
        let chunk_org_z = ctx.cz as f32 * sz;

        let mut rng = XorShift64::new(self.params.seed ^ hash2(ctx.cx, ctx.cz));

        let mut out = Vec::with_capacity(
            self.params.blocks_per_chunk_x * self.params.blocks_per_chunk_z
            * self.params.lots_x * self.params.lots_z
        );

        for bxi in 0..self.params.blocks_per_chunk_x {
            for bzi in 0..self.params.blocks_per_chunk_z {
                let major_x = self.params.major_every > 0 && (bxi % self.params.major_every == 0);
                let major_z = self.params.major_every > 0 && (bzi % self.params.major_every == 0);
                if major_x || major_z { continue; }

                let mut block_x = -0.5*sx + bxi as f32 * bx + self.params.road_w_minor * 0.5;
                let mut block_z = -0.5*sz + bzi as f32 * bz + self.params.road_w_minor * 0.5;
                if (bxi % self.params.major_every) > 0 && ((bxi / self.params.major_every) > 0) {
                    block_x += (self.params.road_w_major - self.params.road_w_minor) * ((bxi / self.params.major_every) as f32);
                }
                if (bzi % self.params.major_every) > 0 && ((bzi / self.params.major_every) > 0) {
                    block_z += (self.params.road_w_major - self.params.road_w_minor) * ((bzi / self.params.major_every) as f32);
                }

                for lx in 0..self.params.lots_x {
                    for lz in 0..self.params.lots_z {
                        let x = chunk_org_x + block_x + (lx as f32) * (self.params.lot_w + self.params.lot_gap) + self.params.lot_w * 0.5;
                        let z = chunk_org_z + block_z + (lz as f32) * (self.params.lot_d + self.params.lot_gap) + self.params.lot_d * 0.5;

                        let (w_low, w_high, w_land) = self.zone_weights(x, z);
                        let pick = rng.unit_f32();
                        let cat = if pick < w_low {
                            BuildingCategory::Lowrise
                        } else if pick < (w_low + w_high) {
                            BuildingCategory::Highrise
                        } else {
                            BuildingCategory::Landmark
                        };

                        let archetype_id = Self::pick_archetype(assets, cat, &mut rng).unwrap_or(0) as u16;

                        let sx = 0.85 + 0.30 * rng.unit_f32();
                        let sz = 0.85 + 0.30 * rng.unit_f32();
                        let sy = match cat {
                            BuildingCategory::Lowrise  => 0.8 + 0.7 * rng.unit_f32(),
                            BuildingCategory::Highrise => {
                                let boost = (1.0 + 1.2 * (1.0 - (x.hypot(z) / 1000.0)).clamp(0.0, 1.0));
                                (0.8 + 1.7 * rng.unit_f32()) * boost
                            }
                            BuildingCategory::Landmark => 0.8 + 0.8 * rng.unit_f32(),
                        };

                        let base_half = assets.base_half(archetype_id as usize);
                        let center_y = base_half.y * sy;

                        out.push(Placement {
                            center: Vector3::new(x, center_y, z),
                            scale:  Vector3::new(sx, sy, sz),
                            archetype_id,
                        });
                    }
                }
            }
        }

        out
    }
}

// -------------- ML Designer (optional, falls back to rule) --------------

pub struct MLDesigner {
    pub fallback: RuleDesigner,
    cat_bias: [f32; 3],
    ml_enabled: bool,
}

impl MLDesigner {
    pub fn new(params: crate::mesh::CityGenParams) -> Self {
        Self {
            fallback: RuleDesigner { params },
            cat_bias: [0.0, 0.0, 0.0],
            ml_enabled: false,
        }
    }

    #[cfg(all(not(target_arch="wasm32"), feature="ml_native"))]
    pub fn load_native_onnx(&mut self, _bytes: &[u8]) {
        self.ml_enabled = true;
    }

    #[cfg(all(target_arch="wasm32", feature="ml_web"))]
    pub fn load_web_model(&mut self, _bytes: &[u8]) {
        self.ml_enabled = true;
    }

    fn apply_bias(&self, (mut low, mut high, mut land): (f32,f32,f32)) -> (f32,f32,f32) {
        low  = (low  + self.cat_bias[0]).max(0.0);
        high = (high + self.cat_bias[1]).max(0.0);
        land = (land + self.cat_bias[2]).max(0.0);
        let sum = (low + high + land).max(1e-5);
        (low/sum, high/sum, land/sum)
    }
}

impl CityDesigner for MLDesigner {
    fn design_chunk(&mut self, ctx: &DesignContext, assets: &AssetLibrary) -> Vec<Placement> {
        // For now: reuse rule-based but bias category probabilities.
        let params = self.fallback.params;
        let (bx, bz) = crate::mesh::block_world_span(&params);
        let (sx, sz) = crate::mesh::chunk_world_span(&params);
        let chunk_org_x = ctx.cx as f32 * sx;
        let chunk_org_z = ctx.cz as f32 * sz;

        let mut rng = XorShift64::new(params.seed ^ hash2(ctx.cx, ctx.cz));
        let mut out = Vec::with_capacity(
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

                        let (w_low, w_high, w_land) = self.fallback.zone_weights(x, z);
                        let (w_low, w_high, w_land) = self.apply_bias((w_low, w_high, w_land));

                        let pick = rng.unit_f32();
                        let cat = if pick < w_low {
                            BuildingCategory::Lowrise
                        } else if pick < (w_low + w_high) {
                            BuildingCategory::Highrise
                        } else {
                            BuildingCategory::Landmark
                        };

                        let ids = assets.indices_by_category(cat);
                        if ids.is_empty() { continue; }
                        let k = (rng.next() as usize) % ids.len();
                        let archetype_id = ids[k] as u16;

                        let sx = 0.85 + 0.30 * rng.unit_f32();
                        let sz = 0.85 + 0.30 * rng.unit_f32();
                        let sy = match cat {
                            BuildingCategory::Lowrise  => 0.8 + 0.7 * rng.unit_f32(),
                            BuildingCategory::Highrise => (0.8 + 1.7 * rng.unit_f32()) * (1.0 + 1.2 * (1.0 - (x.hypot(z) / 1000.0)).clamp(0.0, 1.0)),
                            BuildingCategory::Landmark => 0.8 + 0.8 * rng.unit_f32(),
                        };

                        let base_half = assets.base_half(archetype_id as usize);
                        let center_y = base_half.y * sy;

                        out.push(Placement {
                            center: Vector3::new(x, center_y, z),
                            scale:  Vector3::new(sx, sy, sz),
                            archetype_id,
                        });
                    }
                }
            }
        }

        out
    }
}

// Small hash for (cx,cz)
pub(crate) fn hash2(a: i32, b: i32) -> u64 {
    let mut x = (a as i64 as i128) as u128 ^ (((b as i64 as i128) << 1) as u128) ^ 0x9E37_79B9_7F4A_7C15u128;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u128);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu128);
    (x ^ (x >> 31)) as u64
}
