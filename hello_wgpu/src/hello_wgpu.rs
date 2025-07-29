use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Vector3}; // SquareMatrix gives Matrix4::identity()
use instant::Instant;
use log::{trace, debug, info, warn, error};
use wgpu::util::DeviceExt;
use wgpu::StoreOp;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
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
use crate::mesh;

// -------- Uniforms & Instances ----------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

// Pad the uniform buffer allocation to 256 bytes for maximum backend compatibility.
const CAMERA_BUFFER_SIZE: u64 = 256;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceRaw {
    model: [[f32; 4]; 4],
}

fn mat4_to_array(m: &Matrix4<f32>) -> [[f32; 4]; 4] {
    [
        [m.x.x, m.x.y, m.x.z, m.x.w],
        [m.y.x, m.y.y, m.y.z, m.y.w],
        [m.z.x, m.z.y, m.z.z, m.z.w],
        [m.w.x, m.w.y, m.w.z, m.w.w],
    ]
}

// ---------- Logging Init ----------

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
        let env = Env::default().filter_or(
            "RUST_LOG",
            "hello_wgpu=trace,wgpu_core=warn,wgpu_hal=warn",
        );
        let _ = Builder::from_env(env).try_init();
        eprintln!("Logging initialized (native)");
    }

    info!("init_logging: is_web={}", is_web);
}

pub async fn run(is_web: bool) {
    init_logging(is_web);

    info!("Creating event loop");
    let event_loop = EventLoop::new().expect("failed to create EventLoop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(is_web);
    info!("Running app");
    if let Err(e) = event_loop.run_app(&mut app) {
        error!("Event loop error: {e:?}");
    }
}

struct Engine {
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,

    // Depth
    depth_format: wgpu::TextureFormat,
    depth_view:   wgpu::TextureView,

    // Camera resources
    camera_bgl: wgpu::BindGroupLayout,
    camera_bg:  wgpu::BindGroup,
    camera_buf: wgpu::Buffer,

    // Mesh and instances
    cube_mesh:      mesh::Mesh,
    instance_buf:   wgpu::Buffer,
    instance_count: u32,
}

impl Engine {
    fn update_camera(&self, vp: &Matrix4<f32>) {
        trace!("update_camera: uploading VP matrix");
        let data = CameraUniform { view_proj: mat4_to_array(vp) };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&data));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        trace!("render: acquiring current texture");
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        trace!("render: creating command encoder");
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") }
        );

        trace!("render: begin render pass");
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.05, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.camera_bg, &[]);
            rpass.set_vertex_buffer(0, self.cube_mesh.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, self.instance_buf.slice(..));
            rpass.set_index_buffer(self.cube_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.cube_mesh.index_count, 0, 0..self.instance_count);
        }

        trace!("render: submitting queue");
        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn recreate_depth(&mut self, width: u32, height: u32) {
        info!("recreate_depth: {}x{}", width, height);
        let desc = wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        };
        let depth_tex = self.device.create_texture(&desc);
        self.depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            warn!("resize: skipped (zero dimension) {}x{}", new_size.width, new_size.height);
            return;
        }
        info!("resize: {}x{}", new_size.width, new_size.height);
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.recreate_depth(new_size.width, new_size.height);
    }
}

struct App {
    is_web: bool,
    window: Option<Window>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    engine: Option<Engine>,

    // input/camera
    keyboard_input: camera::KeyboardInput,
    camera: camera::Camera,
    last_cursor_pos: Option<PhysicalPosition<f64>>,

    // frame limiter
    last_frame_time: Instant,

    // async setup
    engine_ready: Arc<AtomicBool>,
    pending_gpu: Arc<Mutex<Option<(wgpu::Device, wgpu::Queue)>>>,
    pending_adapter: Arc<Mutex<Option<wgpu::Adapter>>>,
    instance: Option<wgpu::Instance>,
}

impl App {
    fn new(is_web: bool) -> Self {
        info!("App::new(is_web={})", is_web);
        Self {
            is_web,
            window: None,
            surface: None,
            adapter: None,
            engine: None,
            keyboard_input: camera::KeyboardInput::new(),
            camera: camera::Camera::new(),
            last_cursor_pos: None,
            last_frame_time: Instant::now(),
            engine_ready: Arc::new(AtomicBool::new(false)),
            pending_gpu: Arc::new(Mutex::new(None)),
            pending_adapter: Arc::new(Mutex::new(None)),
            instance: None,
        }
    }

    async fn build_device_queue(
        adapter: wgpu::Adapter,
        out_slot: Arc<Mutex<Option<(wgpu::Device, wgpu::Queue)>>>,
        ready: Arc<AtomicBool>,
    ) {
        info!("build_device_queue: requesting device from adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();

        device.on_uncaptured_error(Box::new(|e| {
            error!("WGPU Uncaptured Error: {e:?}");
        }));

        {
            let mut slot = out_slot.lock().unwrap();
            *slot = Some((device, queue));
        }
        ready.store(true, Ordering::SeqCst);
        info!("build_device_queue: device/queue ready");
    }

    fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { shader_location: 2, offset: 0,  format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 3, offset: 16, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 4, offset: 32, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 5, offset: 48, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }

    fn finalize_engine(&mut self) {
        // These traces help confirm finalize timing:
        trace!("finalize_engine: called, ready_flag={}", self.engine_ready.load(Ordering::SeqCst));

        if self.engine.is_some() {
            trace!("finalize_engine: already created");
            return;
        }
        if !self.engine_ready.load(Ordering::SeqCst) {
            trace!("finalize_engine: not ready yet");
            return;
        }

        info!("finalize_engine: starting");
        let (device, queue) = {
            let mut slot = self.pending_gpu.lock().unwrap();
            slot.take().expect("ready flag set but no device/queue stored")
        };

        let mut surface = self.surface.take().expect("surface missing");

        let adapter: wgpu::Adapter = if let Some(a) = &self.adapter {
            a.clone()
        } else {
            let mut slot = self.pending_adapter.lock().unwrap();
            let a = slot.take().expect("adapter not ready yet (web)");
            self.adapter = Some(a.clone());
            a
        };

        let info = adapter.get_info();
        info!("Adapter: name='{}', backend={:?}", info.name, info.backend);

        let size = self.window.as_ref().unwrap().inner_size();
        let caps = surface.get_capabilities(&adapter);
        debug!("Surface caps: formats={:?}, present_modes={:?}, alpha_modes={:?}",
            caps.formats, caps.present_modes, caps.alpha_modes);

        let format = caps.formats[0];
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };
        info!("Chosen surface format={:?}, alpha={:?}", format, alpha_mode);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };
        surface.configure(&device, &config);
        info!("Surface configured: {}x{}", size.width, size.height);

        // --- Depth setup ---
        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        info!("Depth texture created ({:?})", depth_format);

        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
        });
        info!("Shader module created");

        // --- Camera BGL/BG/Buffer (group=0, binding=0) ---
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64), // mat4x4<f32>
                },
                count: None,
            }],
        });
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Buffer"),
            size: CAMERA_BUFFER_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });
        info!("Camera uniform created (buffer {} bytes)", CAMERA_BUFFER_SIZE);

        // --- Pipeline layout (use camera group at index 0) ---
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

        // --- Pipeline (vertex buffers: mesh + instance) ---
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[mesh::Vertex::layout(), Self::instance_buffer_layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        info!("Render pipeline created");

        // --- Mesh & instances ---
        let cube_mesh = mesh::create_cube(&device);
        info!("Cube mesh buffers ready (indices: {})", cube_mesh.index_count);

        // Grid of instances (10x10)
        let grid = 10usize;
        let mut instance_data = Vec::with_capacity(grid * grid);
        for x in 0..grid {
            for z in 0..grid {
                let model = Matrix4::<f32>::from_translation(Vector3::new(
                    (x as f32) * 2.5,
                    0.0,
                    (z as f32) * 2.5,
                ));
                instance_data.push(InstanceRaw { model: mat4_to_array(&model) });
            }
        }
        let instance_count = instance_data.len() as u32;
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        info!("Instance buffer created ({} instances)", instance_count);

        let mut engine = Engine {
            device,
            queue,
            surface,
            config,
            shader,
            pipeline_layout,
            render_pipeline,
            depth_format,
            depth_view,
            camera_bgl,
            camera_bg,
            camera_buf,
            cube_mesh,
            instance_buf,
            instance_count,
        };
        engine.resize(size);

        self.engine = Some(engine);
        info!("finalize_engine: done");
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        info!("Application resumed");
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
                    canvas.set_width(w);
                    canvas.set_height(h);
                    info!("Canvas size initialized to {}x{}", w, h);
                }
                WindowAttributes::default()
                    .with_title("3D Web Engine")
                    .with_canvas(Some(canvas))
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                panic!("is_web=true but not compiling for wasm32 target");
            }
        } else {
            WindowAttributes::default().with_title("3D Web Engine")
        };

        let window = event_loop.create_window(window_attributes).unwrap();
        info!("Window created");
        self.window = Some(window);

        let backends = if self.is_web {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };
        info!("Using backends: {:?}", backends);

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        self.instance = Some(instance);
        let instance_ref = self.instance.as_ref().unwrap();

        let tempsurface = unsafe {
            instance_ref.create_surface(self.window.as_ref().unwrap()).expect("create_surface failed")
        };
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(tempsurface) };
        self.surface = Some(surface);
        info!("Surface created");

        let ready_flag = Arc::clone(&self.engine_ready);
        let out_slot   = Arc::clone(&self.pending_gpu);
        let adapter_slot = Arc::clone(&self.pending_adapter);

        #[cfg(target_arch = "wasm32")]
        {
            let backends_copy  = backends;

            info!("Requesting adapter (web)");
            wasm_bindgen_futures::spawn_local(async move {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: backends_copy,
                    ..Default::default()
                });

                match instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                {
                    Some(adapter) => {
                        info!("Adapter acquired (web)");
                        { *adapter_slot.lock().unwrap() = Some(adapter.clone()); }
                        App::build_device_queue(adapter, out_slot, ready_flag).await;
                    }
                    None => {
                        error!("No adapter available (web)");
                    }
                }
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            info!("Requesting adapter (native)");
            let adapter = pollster::block_on(instance_ref.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: self.surface.as_ref(),
                    force_fallback_adapter: false,
                },
            ))
            .expect("No compatible adapter found");
            info!("Adapter acquired (native)");
            self.adapter = Some(adapter.clone());

            std::thread::spawn(move || {
                info!("Spawning device/queue creation thread");
                pollster::block_on(async {
                    App::build_device_queue(adapter, out_slot, ready_flag).await;
                });
            });
        }
    }

    fn about_to_wait(&mut self, _el: &ActiveEventLoop) {
        // Keep asking for redraws — this drives the render loop.
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
        if Some(id) != self.window.as_ref().map(|w| w.id()) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                info!("Close requested; exiting");
                event_loop.exit();
            },
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            trace!("key down: {:?}", code);
                            self.keyboard_input.key_press(code)
                        }
                        ElementState::Released => {
                            trace!("key up: {:?}", code);
                            self.keyboard_input.key_release(code)
                        }
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
                info!("Window resized: {}x{}", size.width, size.height);
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size);
                }
            },
            WindowEvent::RedrawRequested => {
                trace!("RedrawRequested");
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                if dt >= 0.016 { // ~60 FPS limiter
                    self.camera.update(dt, &self.keyboard_input);
                    self.last_frame_time = now;

                    self.finalize_engine();

                    if let Some(engine) = self.engine.as_mut() {
                        // Compute VP and upload to the uniform
                        let size = self.window.as_ref().unwrap().inner_size();
                        let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
                        let vp = self.camera.view_projection(aspect);
                        engine.update_camera(&vp);

                        match engine.render() {
                            Ok(()) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                                warn!("Surface lost/outdated; reconfiguring");
                                let size = self.window.as_ref().unwrap().inner_size();
                                engine.resize(size);
                            }
                            Err(wgpu::SurfaceError::Timeout) => {
                                warn!("Surface timeout");
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                error!("Out of memory; exiting");
                                event_loop.exit();
                            }
                            Err(wgpu::SurfaceError::Other) => {
                                error!("Unknown surface error");
                            }
                        }
                    } else {
                        trace!("RedrawRequested: engine not ready yet");
                    }

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            _ => {} // << keep the catch‑all ONLY here, at the end
        }
    }
}
