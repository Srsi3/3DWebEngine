use cgmath::Vector3;
use crate::assets::{AssetLibrary, BuildingCategory};

use crate::chunking::{block_world_span, chunk_world_span, CityGenParams};
#[derive(Copy, Clone)]
pub struct Placement {
    pub center: Vector3<f32>,
    pub scale:  Vector3<f32>,
    pub archetype_id: u16,
}

pub struct DesignContext {
    pub cx: i32,
    pub cz: i32,
    pub seed: u64,
}

pub trait CityDesigner {
    fn design_chunk(&mut self, ctx: &DesignContext, assets: &AssetLibrary) -> Vec<Placement>;
}

// RNG
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self { Self(seed | 1) }
    fn next(&mut self) -> u64 { let mut x=self.0; x^=x<<13; x^=x>>7; x^=x<<17; self.0=x; x }
    fn unit_f32(&mut self) -> f32 { (self.next() as f64 / u64::MAX as f64) as f32 }
}
pub(crate) fn hash2(a: i32, b: i32) -> u64 {
    let mut x = (a as i64 as i128) as u128 ^ (((b as i64 as i128) << 1) as u128) ^ 0x9E37_79B9_7F4A_7C15u128;
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9u128);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EBu128);
    (x ^ (x >> 31)) as u64
}

// ---------------- Rule designer with techno-medieval flavor ----------------

pub struct RuleDesigner {
    pub params: CityGenParams,
}

impl RuleDesigner {
    fn zone_weights(&self, x: f32, z: f32) -> (f32,f32,f32) {
        // Medieval “old town” near center, tech ring farther out.
        let dist = x.hypot(z);
        let old_town = (1.0 - (dist / 900.0)).clamp(0.0, 1.0);
        let tech_ring = ((dist - 300.0) / 700.0).clamp(0.0, 1.0);

        let w_low  = 0.55*old_town + 0.25*(1.0-old_town);
        let w_high = 0.65*tech_ring + 0.10*(1.0-tech_ring);
        let w_land = 0.15 + 0.05*(old_town + tech_ring);
        // normalize
        let s = (w_low + w_high + w_land).max(1e-5);
        (w_low/s, w_high/s, w_land/s)
    }

    fn pick_archetype(assets: &AssetLibrary, cat: BuildingCategory, rng: &mut XorShift64) -> Option<usize> {
        let ids = assets.indices_by_category(cat);
        if ids.is_empty() { return None; }
        let k = (rng.next() as usize) % ids.len();
        Some(ids[k])
    }
}

impl CityDesigner for RuleDesigner {
    fn design_chunk(&mut self, ctx: &DesignContext, assets: &AssetLibrary) -> Vec<Placement> {
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

                        let (mut w_low, mut w_high, mut w_land) = self.zone_weights(x, z);

                        // Occasionally inject a landmark “gate” near grid seams to suggest walls.
                        if ((x / 60.0).sin().abs() < 0.02) || ((z / 60.0).cos().abs() < 0.02) {
                            w_land = (w_land + 0.2).min(0.8);
                        }
                        let s = w_low + w_high + w_land;
                        w_low /= s; w_high /= s; w_land /= s;

                        // category pick
                        let pick = rng.unit_f32();
                        let cat = if pick < w_low {
                            BuildingCategory::Lowrise
                        } else if pick < (w_low + w_high) {
                            BuildingCategory::Highrise
                        } else {
                            BuildingCategory::Landmark
                        };

                        let id = Self::pick_archetype(assets, cat, &mut rng).unwrap_or(0);

                        let sx = 0.85 + 0.35 * rng.unit_f32();
                        let sz = 0.85 + 0.35 * rng.unit_f32();
                        let sy = match cat {
                            BuildingCategory::Lowrise  => 0.8 + 0.7 * rng.unit_f32(),
                            BuildingCategory::Highrise => 1.2 + 1.3 * rng.unit_f32(),
                            BuildingCategory::Landmark => 1.0 + 1.2 * rng.unit_f32(),
                        };

                        let base = assets.base_half(id);
                        let center_y = base.y * sy;

                        out.push(Placement {
                            center: Vector3::new(x, center_y, z),
                            scale:  Vector3::new(sx, sy, sz),
                            archetype_id: id as u16,
                        });
                    }
                }
            }
        }
        out
    }
}
