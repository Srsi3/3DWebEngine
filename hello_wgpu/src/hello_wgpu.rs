use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};

use cgmath::{
    Matrix4, SquareMatrix, Vector3,
    EuclideanSpace, InnerSpace,
};
use instant::Instant;
use log::{info, warn, error};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowAttributes, WindowId};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;
#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowAttributesExtWebSys;

use crate::camera;
use crate::culling;
use crate::mesh;
use crate::types::InstanceRaw;
use crate::render::Engine;
use crate::chunking::{ChunkManager, ViewerId};
use crate::assets::BuildingCategory;
use crate::designer_ml::{RuleDesigner, CityDesigner};

// -------- logging --------

fn init_logging(is_web: bool) {
    #[cfg(target_arch = "wasm32")]
    {
        let _ = console_log::init_with_level(log::Level::Info);
        #[cfg(feature = "console-panic-hook")]
        console_error_panic_hook::set_once();
        web_sys::console::log_1(&"Logging initialized (WASM)".into());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use env_logger::{Builder, Env};
        let env = Env::default().filter_or("RUST_LOG", "hello_wgpu=trace,wgpu_core=warn,wgpu_hal=warn");
        let _ = Builder::from_env(env).try_init();
        eprintln!("Logging initialized (native)");
    }
    info!("init_logging: is_web={}", is_web);
}

// -------- public entry --------

pub async fn run(is_web: bool) {
    init_logging(is_web);

    let event_loop = EventLoop::new().expect("failed to create EventLoop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(is_web);
    if let Err(e) = event_loop.run_app(&mut app) {
        error!("Event loop error: {e:?}");
    }
}

// -------- App --------

struct App {
    is_web: bool,
    window: Option<Window>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    engine:  Option<Engine>,

    // input / camera
    keyboard_input: camera::KeyboardInput,
    camera: camera::Camera,
    last_cursor_pos: Option<PhysicalPosition<f64>>,

    // frame limiter
    last_frame_time: Instant,

    // async
    engine_ready: Arc<AtomicBool>,
    pending_gpu: Arc<Mutex<Option<(wgpu::Device, wgpu::Queue)>>>,
    pending_adapter: Arc<Mutex<Option<wgpu::Adapter>>>,
    instance: Option<wgpu::Instance>,

    // chunking + finite world
    chunk_mgr: ChunkManager,
    viewer_id: ViewerId,
    designer: RuleDesigner,

    // ground
    ground_model: InstanceRaw,

    // floating origin accumulator
    world_origin: cgmath::Vector3<f64>,

    // LOD/cull ranges
    lod0_max: f32,
    lod1_max: f32,
    cull_max: f32,

    // debug
    debug_enabled: bool,
    debug_last_print: Instant,
}

impl App {
    fn new(is_web: bool) -> Self {
        // City params (generation)
        let params = mesh::CityGenParams {
            lots_x: 3, lots_z: 3,
            lot_w: 3.0, lot_d: 3.0, lot_gap: 0.4,
            road_w_minor: 3.0, road_w_major: 8.0, major_every: 6,
            blocks_per_chunk_x: 8, blocks_per_chunk_z: 8,
            seed: 0xC0FF_EE_u64,
        };

        // Finite world bounds (inclusive chunk coords).
        let bounds = (-4, 4, -4, 4);
        let bake_on_miss = true;

        let chunk_mgr = ChunkManager::new(
            params, 3, bounds, bake_on_miss,
            "./city_chunks" // native dir; web uses localStorage instead
        );

        let viewer_id: ViewerId = 1;

        Self {
            is_web,
            window: None, surface: None, adapter: None, engine: None,
            keyboard_input: camera::KeyboardInput::new(),
            camera: camera::Camera::new(),
            last_cursor_pos: None,
            last_frame_time: Instant::now(),
            engine_ready: Arc::new(AtomicBool::new(false)),
            pending_gpu: Arc::new(Mutex::new(None)),
            pending_adapter: Arc::new(Mutex::new(None)),
            instance: None,
            chunk_mgr,
            viewer_id,
            designer: RuleDesigner { params },
            ground_model: InstanceRaw { pos: [0.0, -0.05, 0.0, 0.0], scale: [1.0, 1.0, 1.0, 0.0] },
            world_origin: cgmath::Vector3::new(0.0, 0.0, 0.0),
            lod0_max: 90.0, lod1_max: 190.0, cull_max: 380.0,
            debug_enabled: false,
            debug_last_print: Instant::now(),
        }
    }

    async fn build_device_queue(
        adapter: wgpu::Adapter,
        out_slot: Arc<Mutex<Option<(wgpu::Device, wgpu::Queue)>>>,
        ready: Arc<AtomicBool>,
    ) {
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        device.on_uncaptured_error(Box::new(|e| {
            error!("WGPU Uncaptured Error: {e:?}");
        }));
        { let mut slot = out_slot.lock().unwrap(); *slot = Some((device, queue)); }
        ready.store(true, Ordering::SeqCst);
    }

    fn finalize_engine(&mut self) {
        if self.engine.is_some() || !self.engine_ready.load(Ordering::SeqCst) { return; }

        let (device, queue) = {
            let mut slot = self.pending_gpu.lock().unwrap();
            slot.take().expect("ready flag set but no device/queue stored")
        };

        let surface = self.surface.take().expect("surface missing");
        let adapter = if let Some(a) = &self.adapter { a.clone() } else {
            let mut slot = self.pending_adapter.lock().unwrap();
            let a = slot.take().expect("adapter not ready yet (web)");
            self.adapter = Some(a.clone());
            a
        };

        let size = self.window.as_ref().unwrap().inner_size();
        let engine = Engine::new(device, queue, surface, &adapter, size);
        self.engine = Some(engine);
    }

    // floating origin
    const ORIGIN_SHIFT_DISTANCE: f32 = 500.0;

    fn maybe_shift_origin(&mut self) {
        let cam_pos = self.camera.position.to_vec();
        if cam_pos.magnitude() > Self::ORIGIN_SHIFT_DISTANCE {
            self.shift_world(cam_pos);
        }
    }

    fn shift_world(&mut self, offset: Vector3<f32>) {
        self.chunk_mgr.apply_shift(offset);
        self.camera.position -= offset;
        self.world_origin += cgmath::Vector3::new(offset.x as f64, offset.y as f64, offset.z as f64);
    }
}

// -------- winit plumbing --------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = if self.is_web {
            #[cfg(target_arch = "wasm32")]
            {
                let document = web_sys::window().unwrap().document().unwrap();
                let canvas = document
                    .get_element_by_id("wasm-canvas")
                    .expect("canvas with id 'wasm-canvas' not found")
                    .dyn_into::<web_sys::HtmlCanvasElement>()
                    .unwrap();
                if canvas.width() == 0 || canvas.height() == 0 {
                    let w = canvas.client_width()  as u32;
                    let h = canvas.client_height() as u32;
                    canvas.set_width(w); canvas.set_height(h);
                }
                WindowAttributes::default()
                    .with_title("3D Web Engine")
                    .with_canvas(Some(canvas))
            }
            #[cfg(not(target_arch = "wasm32"))]
            { panic!("is_web=true but not compiling for wasm32"); }
        } else {
            WindowAttributes::default().with_title("3D Web Engine")
        };

        let window = event_loop.create_window(window_attributes).unwrap();
        self.window = Some(window);

        let backends = if self.is_web {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
        self.instance = Some(instance);
        let instance_ref = self.instance.as_ref().unwrap();

        let tempsurface = unsafe {
            instance_ref.create_surface(self.window.as_ref().unwrap()).expect("create_surface failed")
        };
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(tempsurface) };
        self.surface = Some(surface);

        let ready_flag = Arc::clone(&self.engine_ready);
        let out_slot   = Arc::clone(&self.pending_gpu);
        let adapter_slot = Arc::clone(&self.pending_adapter);

        #[cfg(target_arch = "wasm32")]
        {
            let backends_copy  = backends;
            wasm_bindgen_futures::spawn_local(async move {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends: backends_copy, ..Default::default() });
                let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }).await.expect("request_adapter failed on web");
                { *adapter_slot.lock().unwrap() = Some(adapter.clone()); }
                App::build_device_queue(adapter, out_slot, ready_flag).await;
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapter = pollster::block_on(instance_ref.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: self.surface.as_ref(),
                force_fallback_adapter: false,
            })).expect("No compatible adapter found");
            self.adapter = Some(adapter.clone());
            std::thread::spawn(move || {
                pollster::block_on(async {
                    App::build_device_queue(adapter, out_slot, ready_flag).await;
                });
            });
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if Some(id) != self.window.as_ref().map(|w| w.id()) { return; }

        match event {
            WindowEvent::CloseRequested => { event_loop.exit(); }

            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::KeyCode::*;
                if let PhysicalKey::Code(code) = event.physical_key {
                    let down = matches!(event.state, ElementState::Pressed);
                    if down {
                        match code {
                            KeyF2 => {
                                let s = self.camera.speed;
                                self.camera.speed = if s < 5.0 { 10.0 } else if s < 20.0 { 40.0 } else if s < 80.0 { 2.5 } else { 5.0 };
                                info!("Speed => {}", self.camera.speed);
                            }
                            KeyF3 => { self.debug_enabled = !self.debug_enabled; info!("Debug => {}", self.debug_enabled); }
                            Digit1 => { self.lod0_max = (self.lod0_max - 10.0).max(20.0); info!("LOD0_MAX => {:.1}", self.lod0_max); }
                            Digit2 => { self.lod0_max += 10.0; info!("LOD0_MAX => {:.1}", self.lod0_max); }
                            Digit3 => { self.lod1_max = (self.lod1_max - 10.0).max(self.lod0_max + 10.0); info!("LOD1_MAX => {:.1}", self.lod1_max); }
                            Digit4 => { self.lod1_max += 10.0; info!("LOD1_MAX => {:.1}", self.lod1_max); }
                            Digit5 => { self.cull_max = (self.cull_max - 20.0).max(self.lod1_max + 20.0); info!("CULL_MAX => {:.1}", self.cull_max); }
                            Digit6 => { self.cull_max += 20.0; info!("CULL_MAX => {:.1}", self.cull_max); }
                            BracketLeft  => { self.chunk_mgr.chunk_radius = (self.chunk_mgr.chunk_radius - 1).max(1); info!("Chunk radius => {}", self.chunk_mgr.chunk_radius); }
                            BracketRight => { self.chunk_mgr.chunk_radius = (self.chunk_mgr.chunk_radius + 1).min(8); info!("Chunk radius => {}", self.chunk_mgr.chunk_radius); }
                            KeyR => { // reseed & clear loaded (store persists)
                                self.chunk_mgr.params.seed ^= (Instant::now().elapsed().as_nanos() as u64);
                                self.chunk_mgr.loaded.clear();
                                info!("World reseeded; chunks cleared (store not cleared).");
                            }
                            _ => {}
                        }
                    }
                    match event.state {
                        ElementState::Pressed => self.keyboard_input.key_press(code),
                        ElementState::Released => self.keyboard_input.key_release(code),
                    }
                }
            }

            WindowEvent::CursorMoved { position, .. } => {
                if let Some(last) = self.last_cursor_pos.replace(position) {
                    let dx = (position.x - last.x) as f32;
                    let dy = (position.y - last.y) as f32;
                    self.camera.process_mouse_delta(dx, dy, 0.002);
                } else {
                    self.last_cursor_pos = Some(position);
                }
            }

            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() { engine.resize(size); }
            }

            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                if dt >= 0.016 {
                    self.camera.update(dt, &self.keyboard_input);
                    self.maybe_shift_origin(); // floating origin
                    self.last_frame_time = now;
                    self.finalize_engine();

                    if let Some(engine) = self.engine.as_mut() {
                        // ---- everything stays INSIDE this single &mut borrow of engine ----
                        let assets = engine.assets_ref();

                        // ensure chunks for camera (multiplayer-friendly path)
                        self.chunk_mgr.set_viewer(self.viewer_id, self.camera.position.x, self.camera.position.z);
                        self.chunk_mgr.ensure_for_viewers(&mut self.designer, assets);

                        // VP
                        let size = self.window.as_ref().unwrap().inner_size();
                        let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
                        let vp = self.camera.view_projection(aspect);
                        engine.update_camera(&vp);

                        // cull + LOD (compact instances)
                        let fr = culling::frustum_from_vp(&vp);
                        let cam_pos = self.camera.position.to_vec();

                        let mut v0_low  = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v0_high = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v0_pyr  = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v1_low  = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v1_high = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v1_pyr  = Vec::<InstanceRaw>::with_capacity(4096);
                        let mut v2_bill = Vec::<InstanceRaw>::with_capacity(4096);

                        let mut stats_visible = 0usize;
                        for list in self.chunk_mgr.loaded.values() {
                            for p in list {
                                let dist = (p.center - cam_pos).magnitude();
                                if dist > self.cull_max { continue; }

                                let base_half = assets.base_half(p.archetype_id as usize);
                                let half = cgmath::Vector3::new(
                                    base_half.x * p.scale.x,
                                    base_half.y * p.scale.y,
                                    base_half.z * p.scale.z,
                                );
                                if !culling::aabb_intersects_frustum(p.center, half, &fr) { continue; }
                                stats_visible += 1;

                                if dist <= self.lod0_max {
                                    let raw = InstanceRaw {
                                        pos:   [p.center.x, p.center.y, p.center.z, 0.0],
                                        scale: [p.scale.x,  p.scale.y,  p.scale.z,  0.0],
                                    };
                                    match assets.category_of(p.archetype_id as usize) {
                                        BuildingCategory::Lowrise  => v0_low.push(raw),
                                        BuildingCategory::Highrise => v0_high.push(raw),
                                        BuildingCategory::Landmark => v0_pyr.push(raw),
                                    }
                                } else if dist <= self.lod1_max {
                                    let raw = InstanceRaw {
                                        pos:   [p.center.x, p.center.y, p.center.z, 0.0],
                                        scale: [p.scale.x,  p.scale.y,  p.scale.z,  0.0],
                                    };
                                    match assets.category_of(p.archetype_id as usize) {
                                        BuildingCategory::Lowrise  => v1_low.push(raw),
                                        BuildingCategory::Highrise => v1_high.push(raw),
                                        BuildingCategory::Landmark => v1_pyr.push(raw),
                                    }
                                } else {
                                    // billboard footprint from half
                                    let raw = InstanceRaw {
                                        pos:   [p.center.x, p.center.y, p.center.z, 0.0],
                                        scale: [half.x.max(0.5), (half.y*2.0).max(0.5), 1.0, 0.0],
                                    };
                                    v2_bill.push(raw);
                                }
                            }
                        }

                        engine.update_instances(
                            &v0_low, &v0_high, &v0_pyr,
                            &v1_low, &v1_high, &v1_pyr,
                            &v2_bill,
                            &self.ground_model,
                        );

                        if self.debug_enabled && self.debug_last_print.elapsed().as_secs_f32() > 1.0 {
                            info!(
                                "Chunks: {} | Visible:{}  L0 {}+{}+{}  L1 {}+{}+{}  L2 {} | cam=({:.1},{:.1},{:.1}) origin=({:.0},{:.0},{:.0})",
                                self.chunk_mgr.loaded.len(), stats_visible,
                                v0_low.len(), v0_high.len(), v0_pyr.len(),
                                v1_low.len(), v1_high.len(), v1_pyr.len(),
                                v2_bill.len(),
                                self.camera.position.x, self.camera.position.y, self.camera.position.z,
                                self.world_origin.x, self.world_origin.y, self.world_origin.z,
                            );
                            self.debug_last_print = Instant::now();
                        }

                        match engine.render() {
                            Ok(()) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                let size = self.window.as_ref().unwrap().inner_size();
                                engine.resize(size);
                            }
                            Err(wgpu::SurfaceError::Timeout) => { warn!("Surface timeout"); }
                            Err(wgpu::SurfaceError::OutOfMemory) => { error!("Out of memory"); event_loop.exit(); }
                            Err(wgpu::SurfaceError::Other) => { error!("Unknown surface error"); }
                        }
                    }

                    if let Some(w) = &self.window { w.request_redraw(); }
                }
            }
            _ => {}
        }
    }
}
