use wasm_bindgen::prelude::*;
pub mod hello_wgpu;
pub mod mesh;
pub mod camera;
pub mod culling;
pub mod types;
pub mod render;
pub mod chunking;
pub mod city_store;
pub mod assets;
pub mod designer_ml;
pub mod net_mutations;
pub use hello_wgpu::run; 
cfg_if::cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
      #[wasm_bindgen(start)]
      pub async fn start() { 
        //console_error_panic_hook::set_once();
        hello_wgpu::run(true).await; }
  } else {
      pub fn main() { pollster::block_on(hello_wgpu::run(false)); }
  }
}

