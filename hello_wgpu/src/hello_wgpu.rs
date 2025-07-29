use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use cgmath::{Matrix4, SquareMatrix, Vector3}; // <-- SquareMatrix gives Matrix4::identity()
use instant::Instant;
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

// --- Camera uniform (mat4x4<f32> = 64 bytes) ---
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

// --- Per-instance model matrix (mat4x4<f32>) ---
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

pub async fn run(is_web: bool) {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(is_web);
    event_loop.run_app(&mut app).unwrap();
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
        let data = CameraUniform { view_proj: mat4_to_array(vp) };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&data));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?; // may error (Lost/Outdated/Timeout/OutOfMemory)
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None, // None for 2D color target
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
            rpass.set_vertex_buffer(1, self.instance_buf.slice(..)); // per-instance model matrices
            rpass.set_index_buffer(self.cube_mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.draw_indexed(0..self.cube_mesh.index_count, 0, 0..self.instance_count);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn recreate_depth(&mut self, width: u32, height: u32) {
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
        if new_size.width == 0 || new_size.height == 0 { return; }
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
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap();
        {
            let mut slot = out_slot.lock().unwrap();
            *slot = Some((device, queue));
        }
        ready.store(true, Ordering::SeqCst);
    }

    fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        // 4 x vec4<f32> rows (64 bytes)
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // shader locations 2..5 must match WGSL
                wgpu::VertexAttribute { shader_location: 2, offset: 0,  format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 3, offset: 16, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 4, offset: 32, format: wgpu::VertexFormat::Float32x4 },
                wgpu::VertexAttribute { shader_location: 5, offset: 48, format: wgpu::VertexFormat::Float32x4 },
            ],
        }
    }

    fn finalize_engine(&mut self) {
        if self.engine.is_some() || !self.engine_ready.load(Ordering::SeqCst) {
            return;
        }

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

        let size = self.window.as_ref().unwrap().inner_size();
        let caps = surface.get_capabilities(&adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };
        surface.configure(&device, &config);

        // --- Depth setup ---
        let depth_format = wgpu::TextureFormat::Depth24Plus; // works on native and WebGL2
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

        // --- Shader ---
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
        });

        // --- Camera BGL/BG/Buffer (group=0, binding=0) ---
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<CameraUniform>() as u64),
                },
                count: None,
            }],
        });

        let cam_init = CameraUniform { view_proj: Matrix4::<f32>::identity().into() };
        let camera_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::bytes_of(&cam_init),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

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
                depth_compare: wgpu::CompareFunction::Less, // standard depth test
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // --- Mesh & instances ---
        let cube_mesh = mesh::create_cube(&device);

        // Build a grid of instance transforms (e.g., 10x10)
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
        #[cfg(target_arch = "wasm32")]
        web_sys::console::log_1(&"Engine finalized".into());
    }
}

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
                    canvas.set_width(w);
                    canvas.set_height(h);
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
        self.window = Some(window);

        let backends = if self.is_web {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        self.instance = Some(instance);
        let instance_ref = self.instance.as_ref().unwrap();

        let tempsurface = unsafe { instance_ref.create_surface(self.window.as_ref().unwrap()).unwrap() };
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(tempsurface) };
        self.surface = Some(surface);

        let ready_flag = Arc::clone(&self.engine_ready);
        let out_slot   = Arc::clone(&self.pending_gpu);
        let adapter_slot = Arc::clone(&self.pending_adapter);

        #[cfg(target_arch = "wasm32")]
        {
            let backends_copy  = backends;

            wasm_bindgen_futures::spawn_local(async move {
                let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
                    backends: backends_copy,
                    ..Default::default()
                });

                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter: false,
                    })
                    .await
                    .expect("no adapter");
                { *adapter_slot.lock().unwrap() = Some(adapter.clone()); }
                App::build_device_queue(adapter, out_slot, ready_flag).await;
            });
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            let adapter = pollster::block_on(instance_ref.request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: self.surface.as_ref(),
                    force_fallback_adapter: false,
                },
            ))
            .expect("No compatible adapter found");
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
        if Some(id) != self.window.as_ref().map(|w| w.id()) {
            return;
        }
        match event {
            WindowEvent::CloseRequested => {
                println!("The close button was pressed; stopping");
                event_loop.exit();
            },
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
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
            _ => {},
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size);
                }
            },
            WindowEvent::RedrawRequested => {
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
                                let size = self.window.as_ref().unwrap().inner_size();
                                engine.resize(size);
                            }
                            Err(wgpu::SurfaceError::Timeout) => {
                                eprintln!("Surface timeout");
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                eprintln!("Out of memory");
                                event_loop.exit();
                            }
                            Err(wgpu::SurfaceError::Other) => {
                                eprintln!("Unknown surface error");
                            }
                        }
                    }

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            _ => (),
        }
    }
}
