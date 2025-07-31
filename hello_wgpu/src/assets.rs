//! Asset library – holds archetype metadata and the shared meshes that the
//! renderer batches by category.  You can plug in real geometry later;
//! the current placeholder meshes come from `mesh::*` helpers.

use cgmath::Vector3;

use crate::mesh;

// ───────────────────────── Categories & lookup ──────────────────────────
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BuildingCategory { Lowrise, Highrise, Landmark }

/// Which shared draw-mesh the renderer uses (one VA per enum value)
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum CategoryMesh {
    Lowrise,
    Highrise,
    Landmark,
    Billboard,
    Ground,
}

/// One archetype entry in the table
#[derive(Clone)]
pub struct Archetype {
    pub name: &'static str,
    pub category: BuildingCategory,
    pub base_half: Vector3<f32>,          // for culling / billboard footprint
    pub mesh: Option<mesh::Mesh>,         // None ⇒ use category rep mesh
    pub rep_category_mesh: CategoryMesh,  // which shared VA to draw
}

// ───────────────────────── AssetLibrary struct ─────────────────────────
pub struct AssetLibrary {
    pub archetypes: Vec<Archetype>,
    idx_lowrise:  Vec<usize>,
    idx_highrise: Vec<usize>,
    idx_landmark: Vec<usize>,

    // shared meshes (one VA per category + billboard + ground)
    pub mesh_lowrise:   mesh::Mesh,
    pub mesh_highrise:  mesh::Mesh,
    pub mesh_landmark:  mesh::Mesh,
    pub mesh_billboard: mesh::Mesh,
    pub mesh_ground:    mesh::Mesh,
}

impl AssetLibrary {
    pub fn new(device: &wgpu::Device) -> Self {
        // ---------- shared representative meshes ----------
        let mesh_lowrise   = mesh::make_timber_gable(device);
        let mesh_highrise  = mesh::make_block_tower(device);
        let mesh_landmark  = mesh::make_pyramid(device);
        let mesh_billboard = mesh::make_billboard(device);
        let mesh_ground    = mesh::make_ground_plane(device, 512.0);

        // ---------- optional per-archetype mesh ----------
        let timber_alt_mesh = mesh::make_timber_gable_alt(device);

        // ---------- build archetype table ----------
        let mut archetypes = Vec::<Archetype>::new();
        let mut idx_low = Vec::<usize>::new();
        let mut idx_high= Vec::<usize>::new();
        let mut idx_land= Vec::<usize>::new();

        // helper closure
        let mut push = |name:&'static str,
                        category:BuildingCategory,
                        half:Vector3<f32>,
                        mesh_opt:Option<mesh::Mesh>,
                        rep:CategoryMesh,
                        catlist:&mut Vec<usize>| {
            archetypes.push(Archetype{ name, category, base_half:half,
                                       mesh:mesh_opt, rep_category_mesh:rep});
            catlist.push(archetypes.len()-1);
        };

        // ---- Low-rise variants ----
        let h_low = Vector3::new(0.9,0.9,0.9);
        push("timber_house_a", BuildingCategory::Lowrise, h_low, None,
             CategoryMesh::Lowrise, &mut idx_low);
        push("timber_house_b", BuildingCategory::Lowrise, h_low,
             Some(timber_alt_mesh), CategoryMesh::Lowrise, &mut idx_low);
        push("workshop_neon" , BuildingCategory::Lowrise, h_low, None,
             CategoryMesh::Lowrise, &mut idx_low);

        // ---- High-rise variants ----
        let h_high = Vector3::new(0.7,1.6,0.7);
        push("block_tower_a", BuildingCategory::Highrise, h_high, None,
             CategoryMesh::Highrise, &mut idx_high);
        push("block_tower_b", BuildingCategory::Highrise, h_high, None,
             CategoryMesh::Highrise, &mut idx_high);
        let h_cyl = Vector3::new(0.55,1.5,0.55);
        push("cyl_tower_12", BuildingCategory::Highrise, h_cyl, None,
             CategoryMesh::Highrise, &mut idx_high);

        // ---- Landmarks ----
        let h_pyr = Vector3::new(1.2,1.2,1.2);
        push("pyramid_citadel", BuildingCategory::Landmark, h_pyr, None,
             CategoryMesh::Landmark, &mut idx_land);
        let h_gate = Vector3::new(1.1,1.1,0.8);
        push("gate_arch", BuildingCategory::Landmark, h_gate, None,
             CategoryMesh::Landmark, &mut idx_land);

        Self {
            archetypes,
            idx_lowrise:  idx_low,
            idx_highrise: idx_high,
            idx_landmark: idx_land,
            mesh_lowrise, mesh_highrise, mesh_landmark, mesh_billboard, mesh_ground,
        }
    }

    // ---------- quick lookups ----------
    #[inline] pub fn base_half(&self, id: usize) -> Vector3<f32> {
        self.archetypes[id].base_half
    }
    #[inline] pub fn category_of(&self, id: usize) -> BuildingCategory {
        self.archetypes[id].category
    }
    #[inline] pub fn mesh_of(&self, id: usize) -> Option<&mesh::Mesh> {
        self.archetypes[id].mesh.as_ref()
    }
    #[inline] pub fn indices_by_category(&self, cat: BuildingCategory) -> &[usize] {
        match cat {
            BuildingCategory::Lowrise  => &self.idx_lowrise,
            BuildingCategory::Highrise => &self.idx_highrise,
            BuildingCategory::Landmark => &self.idx_landmark,
        }
    }
    #[inline] pub fn mesh_for(&self, cm: CategoryMesh) -> &mesh::Mesh {
        match cm {
            CategoryMesh::Lowrise   => &self.mesh_lowrise,
            CategoryMesh::Highrise  => &self.mesh_highrise,
            CategoryMesh::Landmark  => &self.mesh_landmark,
            CategoryMesh::Billboard => &self.mesh_billboard,
            CategoryMesh::Ground    => &self.mesh_ground,
        }
    }
}
