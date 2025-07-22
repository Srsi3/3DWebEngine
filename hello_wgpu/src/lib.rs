use wasm_bindgen::prelude::*;
mod hello_wgpu;

cfg_if::cfg_if! {
  if #[cfg(target_arch = "wasm32")] {
      #[wasm_bindgen(start)]
      pub async fn start() { hello_wgpu::run(true).await; }
  } else {
      pub fn main() { pollster::block_on(hello_wgpu::run(false)); }
  }
}

