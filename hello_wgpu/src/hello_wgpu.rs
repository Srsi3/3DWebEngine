use std::time::{Instant, Duration};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
use wgpu::StoreOp;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;

#[cfg(target_arch = "wasm32")]
use web_sys::HtmlCanvasElement;
#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowAttributesExtWebSys;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

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
}

impl Engine {
    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?; // can error
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Triangle Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None, // dont know what this does
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.render_pipeline);
            rpass.draw(0..3, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
    }
}

struct App {
    is_web: bool,
    window: Option<Window>,
    surface: Option<wgpu::Surface<'static>>, 
    adapter: Option<wgpu::Adapter>,
    engine: Option<Engine>,
    last_frame_time: Instant,  //framelimiter
    engine_ready: Arc<AtomicBool>,
    pending_gpu: Arc<Mutex<Option<(wgpu::Device, wgpu::Queue)>>>,
}

impl App {
    fn new(is_web: bool) -> Self {
        Self { is_web, window: None, surface: None, adapter: None,engine: None, last_frame_time: Instant::now(), engine_ready: Arc::new(AtomicBool::new(false)), pending_gpu: Arc::new(Mutex::new(None)), }
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

    // Initialize the engine asynchronously
    fn finalize_engine(&mut self) {
        if self.engine.is_some() || !self.engine_ready.load(Ordering::SeqCst) {
            return; 
        }
        

        let (device, queue) = {
            let mut slot = self.pending_gpu.lock().unwrap();
            slot.take().expect("ready flag set but no device/queue stored")
        };

        let mut surface = self.surface.take().expect("surface missing");

        let adapter = self.adapter.as_ref().unwrap();

        let size = self.window.as_ref().unwrap().inner_size();
        let caps = surface.get_capabilities(adapter);
        let format = caps.formats[0];

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2, //not sure what this does must check docs
        };
        surface.configure(&device, &config);

        
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Triangle Shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Pipeline Layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
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
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None, // not sure what this does, must check docs
    });

    self.engine = Some(Engine {
            device,
            queue,
            surface, 
            config,
            shader,
            pipeline_layout,
            render_pipeline,
        });
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
        //let size = window.inner_size();
        self.window = Some(window);
        //
        let backends = if self.is_web {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let tempsurface = unsafe { instance.create_surface(self.window.as_ref().unwrap()).unwrap() };
        let surface: wgpu::Surface<'static> = unsafe { std::mem::transmute(tempsurface) };
        self.surface = Some(surface);

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: self.surface.as_ref(),
            force_fallback_adapter: false,
        }))
        .expect("No compatible adapter found");
        self.adapter = Some(adapter.clone());

        //
        let ready_flag = self.engine_ready.clone();
        let out_slot = self.pending_gpu.clone();
        
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                App::build_device_queue(adapter, out_slot, ready_flag).await;
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                pollster::block_on(async {
                    App::build_device_queue(adapter, out_slot, ready_flag).await;
                });
            });
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
            WindowEvent::Resized(size) => {
                if let Some(engine) = self.engine.as_mut() {
                    engine.resize(size);
                }
            },
            WindowEvent::RedrawRequested => {
                // Redraw the application.
                //
                // It's preferable for applications that do not render continuously to render in
                // this event rather than in AboutToWait, since rendering in here allows
                // the program to gracefully handle redraws requested by the OS.

                // Draw.

                // Queue a RedrawRequested event.
                //
                // You only need to call this if you've determined that you need to redraw in
                // applications which do not always need to. Applications that redraw continuously
                // can render here instead.
                let now = Instant::now();
                if now.duration_since(self.last_frame_time) >= Duration::from_millis(16) {
                    self.last_frame_time = now;

                    self.finalize_engine();

                    if let Some(engine) = self.engine.as_mut() {
                        match engine.render() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                            // Reconfigure on resize/outdated
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


