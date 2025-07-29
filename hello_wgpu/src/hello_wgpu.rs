use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
use std::time::Duration;

use bytemuck::{Pod, Zeroable};
use cgmath::{
    Matrix4, SquareMatrix, Vector3,
    EuclideanSpace, InnerSpace, // <- needed for to_vec(), magnitude(), dot()
};
use instant::Instant;
use log::{trace, debug, info, warn, error};
use wgpu::util::DeviceExt;
use wgpu::StoreOp;
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
use crate::mesh;
use crate::culling;

// -------- Uniforms & Instances ----------

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniform {
    view_proj: [[f32; 4]; 4],
}

#[derive(Copy, Clone)]
enum MeshKind {
    Lowrise,  // block_lowrise
    Highrise, // tower_highrise
    Pyramid,  // pyramid_tower
}

#[derive(Copy, Clone)]
struct WorldInstanceCPU {
    model:  Matrix4<f32>,
    center: Vector3<f32>,
    half:   Vector3<f32>,
    kind:   MeshKind,
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

// -------------------- ENGINE --------------------

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

    // Meshes
    mesh_lowrise:   mesh::Mesh, // near/mid
    mesh_highrise:  mesh::Mesh, // near/mid
    mesh_pyramid:   mesh::Mesh, // near/mid
    mesh_billboard: mesh::Mesh, // far

    // Instance buffers (per LOD + per mesh where needed)
    // LOD0 (near)
    instbuf_lod0_lowrise:  wgpu::Buffer,
    instbuf_lod0_highrise: wgpu::Buffer,
    instbuf_lod0_pyramid:  wgpu::Buffer,
    cnt_lod0_lowrise:  u32,
    cnt_lod0_highrise: u32,
    cnt_lod0_pyramid:  u32,

    // LOD1 (mid)
    instbuf_lod1_lowrise:  wgpu::Buffer,
    instbuf_lod1_highrise: wgpu::Buffer,
    instbuf_lod1_pyramid:  wgpu::Buffer,
    cnt_lod1_lowrise:  u32,
    cnt_lod1_highrise: u32,
    cnt_lod1_pyramid:  u32,

    // LOD2 (far) -> billboard for every kind
    instbuf_lod2_billboard: wgpu::Buffer,
    cnt_lod2_billboard: u32,
}

impl Engine {
    fn update_camera(&self, vp: &Matrix4<f32>) {
        let data = CameraUniform { view_proj: mat4_to_array(vp) };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&data));
    }

    fn update_instances(
        &mut self,
        // LOD0
        v0_low: &[InstanceRaw],
        v0_high: &[InstanceRaw],
        v0_pyr: &[InstanceRaw],
        // LOD1
        v1_low: &[InstanceRaw],
        v1_high: &[InstanceRaw],
        v1_pyr: &[InstanceRaw],
        // LOD2
        v2_bill: &[InstanceRaw],
    ) {
        if !v0_low.is_empty()  { self.queue.write_buffer(&self.instbuf_lod0_lowrise,  0, bytemuck::cast_slice(v0_low)); }
        if !v0_high.is_empty() { self.queue.write_buffer(&self.instbuf_lod0_highrise, 0, bytemuck::cast_slice(v0_high)); }
        if !v0_pyr.is_empty()  { self.queue.write_buffer(&self.instbuf_lod0_pyramid,  0, bytemuck::cast_slice(v0_pyr)); }
        if !v1_low.is_empty()  { self.queue.write_buffer(&self.instbuf_lod1_lowrise,  0, bytemuck::cast_slice(v1_low)); }
        if !v1_high.is_empty() { self.queue.write_buffer(&self.instbuf_lod1_highrise, 0, bytemuck::cast_slice(v1_high)); }
        if !v1_pyr.is_empty()  { self.queue.write_buffer(&self.instbuf_lod1_pyramid,  0, bytemuck::cast_slice(v1_pyr)); }
        if !v2_bill.is_empty() { self.queue.write_buffer(&self.instbuf_lod2_billboard,0, bytemuck::cast_slice(v2_bill)); }

        self.cnt_lod0_lowrise  = v0_low.len()  as u32;
        self.cnt_lod0_highrise = v0_high.len() as u32;
        self.cnt_lod0_pyramid  = v0_pyr.len()  as u32;

        self.cnt_lod1_lowrise  = v1_low.len()  as u32;
        self.cnt_lod1_highrise = v1_high.len() as u32;
        self.cnt_lod1_pyramid  = v1_pyr.len()  as u32;

        self.cnt_lod2_billboard = v2_bill.len() as u32;
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.02, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.camera_bg, &[]);

            // ---- LOD0 NEAR ----
            if self.cnt_lod0_lowrise > 0 {
                rpass.set_vertex_buffer(0, self.mesh_lowrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_lowrise.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod0_lowrise.slice(..));
                rpass.draw_indexed(0..self.mesh_lowrise.index_count, 0, 0..self.cnt_lod0_lowrise);
            }
            if self.cnt_lod0_highrise > 0 {
                rpass.set_vertex_buffer(0, self.mesh_highrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_highrise.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod0_highrise.slice(..));
                rpass.draw_indexed(0..self.mesh_highrise.index_count, 0, 0..self.cnt_lod0_highrise);
            }
            if self.cnt_lod0_pyramid > 0 {
                rpass.set_vertex_buffer(0, self.mesh_pyramid.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_pyramid.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod0_pyramid.slice(..));
                rpass.draw_indexed(0..self.mesh_pyramid.index_count, 0, 0..self.cnt_lod0_pyramid);
            }

            // ---- LOD1 MID ----
            if self.cnt_lod1_lowrise > 0 {
                rpass.set_vertex_buffer(0, self.mesh_lowrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_lowrise.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod1_lowrise.slice(..));
                rpass.draw_indexed(0..self.mesh_lowrise.index_count, 0, 0..self.cnt_lod1_lowrise);
            }
            if self.cnt_lod1_highrise > 0 {
                rpass.set_vertex_buffer(0, self.mesh_highrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_highrise.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod1_highrise.slice(..));
                rpass.draw_indexed(0..self.mesh_highrise.index_count, 0, 0..self.cnt_lod1_highrise);
            }
            if self.cnt_lod1_pyramid > 0 {
                rpass.set_vertex_buffer(0, self.mesh_pyramid.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_pyramid.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod1_pyramid.slice(..));
                rpass.draw_indexed(0..self.mesh_pyramid.index_count, 0, 0..self.cnt_lod1_pyramid);
            }

            // ---- LOD2 FAR (billboards) ----
            if self.cnt_lod2_billboard > 0 {
                rpass.set_vertex_buffer(0, self.mesh_billboard.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.mesh_billboard.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1, self.instbuf_lod2_billboard.slice(..));
                rpass.draw_indexed(0..self.mesh_billboard.index_count, 0, 0..self.cnt_lod2_billboard);
            }
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
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.recreate_depth(new_size.width, new_size.height);
    }
}

// -------------------- APP --------------------

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

    // CPU-side instances for culling/LOD
    cpu_instances: Vec<WorldInstanceCPU>,

    // Track how far we've shifted the world (global origin), optional for future use
    world_origin: cgmath::Vector3<f64>,
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
            cpu_instances: Vec::new(),
            world_origin: cgmath::Vector3::new(0.0, 0.0, 0.0),
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

        device.on_uncaptured_error(Box::new(|e| {
            error!("WGPU Uncaptured Error: {e:?}");
        }));

        {
            let mut slot = out_slot.lock().unwrap();
            *slot = Some((device, queue));
        }
        ready.store(true, Ordering::SeqCst);
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
        let alpha_mode = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else {
            caps.alpha_modes[0]
        };

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

        // --- Depth
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

        // --- Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
        });

        // --- Camera uniform
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Camera BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
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

        // --- Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });

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

        // --- Meshes (variety)
        let mesh_lowrise   = mesh::create_block_lowrise(&device);
        let mesh_highrise  = mesh::create_tower_highrise(&device);
        let mesh_pyramid   = mesh::create_pyramid_tower(&device);
        let mesh_billboard = mesh::create_billboard_quad(&device);

        // --- CPU instances with mesh variety
        let grid = 64usize;
        let mut cpu_instances = Vec::with_capacity(grid * grid);

        // half-extents per kind (approx for culling)
        let half_lowrise  = Vector3::new(1.5, 0.4, 1.0);  // 3.0 x 0.8 x 2.0
        let half_highrise = Vector3::new(0.45, 3.0, 0.45); // 0.9 x 6.0 x 0.9
        let half_pyramid  = Vector3::new(1.0, 1.1, 1.0);   // base(1.0) + roof apex

        // simple deterministic "random" from grid coords
        fn pick_kind(x: u32, z: u32) -> MeshKind {
            let mut h = x ^ (z << 16) ^ 0x9E3779B9;
            h = h.wrapping_mul(0x85EBCA6B);
            let r = (h % 100) as u32;
            match r {
                0..=39  => MeshKind::Lowrise,   // 40%
                40..=79 => MeshKind::Highrise,  // 40%
                _       => MeshKind::Pyramid,   // 20%
            }
        }

        for x in 0..grid {
            for z in 0..grid {
                let pos = Vector3::new((x as f32) * 4.0, 0.0, (z as f32) * 4.0);
                let kind = pick_kind(x as u32, z as u32);
                let half = match kind {
                    MeshKind::Lowrise  => half_lowrise,
                    MeshKind::Highrise => half_highrise,
                    MeshKind::Pyramid  => half_pyramid,
                };
                let model = Matrix4::<f32>::from_translation(pos);
                cpu_instances.push(WorldInstanceCPU { model, center: pos, half, kind });
            }
        }
        info!("CPU instances created: {}", cpu_instances.len());

        // Count per kind (to size instance buffers)
        let mut count_low = 0usize;
        let mut count_high = 0usize;
        let mut count_pyr = 0usize;
        for inst in &cpu_instances {
            match inst.kind {
                MeshKind::Lowrise  => count_low += 1,
                MeshKind::Highrise => count_high += 1,
                MeshKind::Pyramid  => count_pyr += 1,
            }
        }
        let stride = std::mem::size_of::<InstanceRaw>() as u64;

        // helper to create buffer sized for n instances
        let make_buf = |label: &str, n: usize| -> wgpu::Buffer {
            let size = (n.max(1) as u64) * stride;
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // LOD0 buffers per kind
        let instbuf_lod0_lowrise  = make_buf("instbuf_lod0_lowrise",  count_low);
        let instbuf_lod0_highrise = make_buf("instbuf_lod0_highrise", count_high);
        let instbuf_lod0_pyramid  = make_buf("instbuf_lod0_pyramid",  count_pyr);
        // LOD1 buffers per kind
        let instbuf_lod1_lowrise  = make_buf("instbuf_lod1_lowrise",  count_low);
        let instbuf_lod1_highrise = make_buf("instbuf_lod1_highrise", count_high);
        let instbuf_lod1_pyramid  = make_buf("instbuf_lod1_pyramid",  count_pyr);
        // LOD2 billboard buffer (one for all)
        let instbuf_lod2_billboard = make_buf("instbuf_lod2_billboard", cpu_instances.len());

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
            mesh_lowrise,
            mesh_highrise,
            mesh_pyramid,
            mesh_billboard,
            instbuf_lod0_lowrise,
            instbuf_lod0_highrise,
            instbuf_lod0_pyramid,
            cnt_lod0_lowrise: 0,
            cnt_lod0_highrise: 0,
            cnt_lod0_pyramid: 0,
            instbuf_lod1_lowrise,
            instbuf_lod1_highrise,
            instbuf_lod1_pyramid,
            cnt_lod1_lowrise: 0,
            cnt_lod1_highrise: 0,
            cnt_lod1_pyramid: 0,
            instbuf_lod2_billboard,
            cnt_lod2_billboard: 0,
        };
        engine.resize(size);

        self.engine = Some(engine);
        self.cpu_instances = cpu_instances;
    }

    // ---------------- Floating Origin ----------------

    const ORIGIN_SHIFT_DISTANCE: f32 = 500.0;

    fn maybe_shift_origin(&mut self) {
        let cam_pos = self.camera.position.to_vec();
        if cam_pos.magnitude() > Self::ORIGIN_SHIFT_DISTANCE {
            self.shift_world(cam_pos);
        }
    }

    fn shift_world(&mut self, offset: Vector3<f32>) {
        info!("Floating origin shift by ({:.1}, {:.1}, {:.1})", offset.x, offset.y, offset.z);

        for inst in &mut self.cpu_instances {
            inst.center -= offset;
            inst.model = Matrix4::<f32>::from_translation(inst.center);
        }

        self.camera.position -= offset;
        self.world_origin += cgmath::Vector3::new(offset.x as f64, offset.y as f64, offset.z as f64);
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
                    .expect("request_adapter failed on web");

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
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size);
                }
            },
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(self.last_frame_time).as_secs_f32();
                if dt >= 0.016 {
                    self.camera.update(dt, &self.keyboard_input);

                    // Floating origin before culling/draw
                    self.maybe_shift_origin();

                    self.last_frame_time = now;
                    self.finalize_engine();

                    if let Some(engine) = self.engine.as_mut() {
                        // Compute VP and upload
                        let size = self.window.as_ref().unwrap().inner_size();
                        let aspect = (size.width.max(1) as f32) / (size.height.max(1) as f32);
                        let vp = self.camera.view_projection(aspect);
                        engine.update_camera(&vp);

                        // --------- CULL + LOD ----------
                        const LOD0_MAX: f32 = 80.0;   // near
                        const LOD1_MAX: f32 = 180.0;  // mid
                        const CULL_MAX: f32 = 350.0;  // far cutoff

                        let fr = culling::frustum_from_vp(&vp);
                        let cam_pos = self.camera.position.to_vec();

                        // Visible lists
                        let mut v0_low: Vec<InstanceRaw>  = Vec::with_capacity(1024);
                        let mut v0_high: Vec<InstanceRaw> = Vec::with_capacity(1024);
                        let mut v0_pyr: Vec<InstanceRaw>  = Vec::with_capacity(1024);

                        let mut v1_low: Vec<InstanceRaw>  = Vec::with_capacity(2048);
                        let mut v1_high: Vec<InstanceRaw> = Vec::with_capacity(2048);
                        let mut v1_pyr: Vec<InstanceRaw>  = Vec::with_capacity(2048);

                        let mut v2_bill: Vec<InstanceRaw> = Vec::with_capacity(4096);

                        for inst in &self.cpu_instances {
                            let dist = (inst.center - cam_pos).magnitude();
                            if dist > CULL_MAX { continue; }
                            if !culling::aabb_intersects_frustum(inst.center, inst.half, &fr) { continue; }

                            let raw = InstanceRaw { model: mat4_to_array(&inst.model) };
                            if dist <= LOD0_MAX {
                                match inst.kind {
                                    MeshKind::Lowrise  => v0_low.push(raw),
                                    MeshKind::Highrise => v0_high.push(raw),
                                    MeshKind::Pyramid  => v0_pyr.push(raw),
                                }
                            } else if dist <= LOD1_MAX {
                                match inst.kind {
                                    MeshKind::Lowrise  => v1_low.push(raw),
                                    MeshKind::Highrise => v1_high.push(raw),
                                    MeshKind::Pyramid  => v1_pyr.push(raw),
                                }
                            } else {
                                v2_bill.push(raw);
                            }
                        }

                        engine.update_instances(
                            &v0_low, &v0_high, &v0_pyr,
                            &v1_low, &v1_high, &v1_pyr,
                            &v2_bill,
                        );

                        // Render
                        match engine.render() {
                            Ok(()) => {}
                            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
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
                    }

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
            }
            _ => {}
        }
    }
}
