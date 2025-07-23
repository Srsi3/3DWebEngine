use std::time::{Instant, Duration};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};
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

        self.engine = Some(Engine {
            device,
            queue,
            surface, 
            config,
        });
    }// TODO: create pipelines, upload assets, etc.
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

                    if let Some(engine) = &self.engine {
                        // TODO: actual rendering:
                        // let frame = match engine.surface.get_current_texture() { ... }
                        // encode commands, submit queue, frame.present()
                        let _ = engine; // silence unused warning for now
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


