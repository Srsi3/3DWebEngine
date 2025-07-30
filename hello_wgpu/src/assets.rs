//! Asset archetype registry.
//! Keeps the engine lightweight by starting with built-in, vertex-colored meshes,
//! while allowing optional glTF loading behind a Cargo feature later.

use crate::mesh;

/// Batching category to keep draw calls low.
/// You can extend this list if you add more pipelines/materials.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BuildingCategory {
    Lowrise,
    Highrise,
    Landmark, // used for pyramid/unique types
}

/// Half-extents (base footprint / height) at scale=1.
/// Used for culling and billboard sizing.
#[derive(Copy, Clone, Debug)]
pub struct BaseHalf {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// One building archetype with LODs and a billboard mesh.
pub struct Archetype {
    pub name: String,
    pub category: BuildingCategory,
    pub lod0: mesh::Mesh,
    pub lod1: mesh::Mesh,
    pub lod2: mesh::Mesh,
    pub billboard: mesh::Mesh,
    pub base_half: BaseHalf,
}

/// Small registry of archetypes.
/// Start with 3 built-in archetypes; add more later (or via optional glTF).
pub struct AssetLibrary {
    archetypes: Vec<Archetype>,
}

impl AssetLibrary {
    /// Create a default library using built-in meshes.
    pub fn new(device: &wgpu::Device) -> Self {
        let builtins = crate::mesh::create_city_meshes(device);

        // NOTE: We create separate meshes for each archetype now to avoid Handle/Clone complexity.
        // With few archetypes this is fine. If you later want to share buffers across archetypes,
        // refactor Mesh into Arc<MeshInner>.
        let lowrise = Archetype {
            name: "lowrise_box".into(),
            category: BuildingCategory::Lowrise,
            lod0: crate::mesh::create_block_lowrise(device),
            lod1: crate::mesh::create_block_lowrise(device),
            lod2: crate::mesh::create_block_lowrise(device),
            billboard: crate::mesh::create_billboard_quad(device),
            base_half: BaseHalf { x: 1.5, y: 0.4, z: 1.0 },
        };

        let highrise = Archetype {
            name: "highrise_box".into(),
            category: BuildingCategory::Highrise,
            lod0: crate::mesh::create_tower_highrise(device),
            lod1: crate::mesh::create_tower_highrise(device),
            lod2: crate::mesh::create_tower_highrise(device),
            billboard: crate::mesh::create_billboard_quad(device),
            base_half: BaseHalf { x: 0.45, y: 3.0, z: 0.45 },
        };

        let landmark = Archetype {
            name: "pyramid_tower".into(),
            category: BuildingCategory::Landmark,
            lod0: crate::mesh::create_pyramid_tower(device),
            lod1: crate::mesh::create_pyramid_tower(device),
            lod2: crate::mesh::create_pyramid_tower(device),
            billboard: crate::mesh::create_billboard_quad(device),
            base_half: BaseHalf { x: 1.0, y: 0.75, z: 1.0 }, // approx base+roof
        };

        let mut lib = Self { archetypes: Vec::with_capacity(8) };
        lib.archetypes.push(lowrise);
        lib.archetypes.push(highrise);
        lib.archetypes.push(landmark);

        // (Optional) Also keep a "ground" mesh in mesh.rs; not an archetype.
        // builtins.ground is used by renderer directly.

        lib
    }

    #[inline]
    pub fn len(&self) -> usize { self.archetypes.len() }

    #[inline]
    pub fn get(&self, idx: usize) -> &Archetype { &self.archetypes[idx] }

    /// Find one archetype index for a requested category (random choice can be done externally).
    pub fn any_index_by_category(&self, cat: BuildingCategory) -> Option<usize> {
        self.archetypes.iter().enumerate().find(|(_, a)| a.category == cat).map(|(i,_)| i)
    }

    /// Return all indices for a category (to randomly pick variants).
    pub fn indices_by_category(&self, cat: BuildingCategory) -> Vec<usize> {
        self.archetypes.iter().enumerate().filter_map(|(i,a)| (a.category==cat).then_some(i)).collect()
    }

    /// Access base half-extents for culling/billboards.
    pub fn base_half(&self, idx: usize) -> BaseHalf {
        self.archetypes[idx].base_half
    }

    /// Map an archetype id to a coarse category (useful for legacy code paths).
    pub fn category_of(&self, idx: usize) -> BuildingCategory {
        self.archetypes[idx].category
    }

    /// Access LOD meshes (used by the renderer set-up).
    pub fn meshes_for(&self, idx: usize) -> (&mesh::Mesh, &mesh::Mesh, &mesh::Mesh, &mesh::Mesh) {
        let a = &self.archetypes[idx];
        (&a.lod0, &a.lod1, &a.lod2, &a.billboard)
    }
}
