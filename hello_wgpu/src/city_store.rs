//! Finite-world chunk persistence.
//! Native: ./city_chunks/{cx}_{cz}.bin (bincode).
//! Web   : window.localStorage["city_chunk_{cx}_{cz}"] = base64(bincode).

use crate::mesh::{BuildingDisk, BuildingRecord};
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
pub struct ChunkFile {
    pub cx: i32,
    pub cz: i32,
    pub buildings: Vec<BuildingDisk>,
}

// ---------- Native FS impl ----------

#[cfg(not(target_arch = "wasm32"))]
pub mod native {
    use super::*;
    use std::fs;
    use std::path::{Path, PathBuf};

    fn dir_path(dir: &str) -> PathBuf {
        Path::new(dir).to_path_buf()
    }
    fn file_path(dir: &str, cx: i32, cz: i32) -> PathBuf {
        dir_path(dir).join(format!("{}_{}.bin", cx, cz))
    }

    pub fn load_chunk(dir: &str, cx: i32, cz: i32) -> Option<ChunkFile> {
        let p = file_path(dir, cx, cz);
        let bytes = fs::read(p).ok()?;
        bincode::deserialize::<ChunkFile>(&bytes).ok()
    }

    pub fn save_chunk(dir: &str, chunk: &ChunkFile) -> std::io::Result<()> {
        let d = dir_path(dir);
        if !d.exists() { std::fs::create_dir_all(&d)?; }
        let p = file_path(dir, chunk.cx, chunk.cz);
        let bytes = bincode::serialize(chunk).expect("bincode serialize");
        std::fs::write(p, bytes)
    }
}

// ---------- Web localStorage impl ----------

#[cfg(target_arch = "wasm32")]
pub mod web {
    use super::*;
    use wasm_bindgen::JsValue;

    fn key(cx: i32, cz: i32) -> String {
        format!("city_chunk_{}_{}", cx, cz)
    }

    pub fn load_chunk(_dir_unused: &str, cx: i32, cz: i32) -> Option<ChunkFile> {
        let window = web_sys::window()?;
        let storage = window.local_storage().ok()??;
        let k = key(cx, cz);
        let s = storage.get_item(&k).ok()??;
        let bytes = base64::decode(s).ok()?;
        bincode::deserialize::<ChunkFile>(&bytes).ok()
    }

    pub fn save_chunk(_dir_unused: &str, chunk: &ChunkFile) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or(JsValue::from_str("no window"))?;
        let storage = window.local_storage()?.ok_or(JsValue::from_str("no localStorage"))?;
        let k = key(chunk.cx, chunk.cz);
        let bytes = bincode::serialize(chunk).map_err(|e| JsValue::from_str(&format!("{e}")))?;
        let s = base64::encode(bytes);
        storage.set_item(&k, &s)
    }
}
