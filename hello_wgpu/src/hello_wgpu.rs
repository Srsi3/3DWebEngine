use std::time::{Instant, Duration};
use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowAttributesExtWeb;


pub async fn run(is_web: bool) {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(is_web);
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    is_web: bool,
    window: Option<Window>,
    last_frame_time: Instant,  //framelimiter
    engine_ready: Arc<AtomicBool>,
}

impl App {
    fn new(is_web: bool) -> Self {
        Self { is_web, window: None, last_frame_time: Instant::now(), engine_ready: Arc::new(AtomicBool::new(false)), }
    }
    // Initialize the engine asynchronously
    async fn init_engine(window: Window, is_web: bool) {
        let backends = if is_web {
            wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
        } else {
            wgpu::Backends::all()
        };

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });

        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("No compatible adapter found (WebGPU+GL)");

        let (_device, _queue) = adapter
            .request_device(&Default::default())
            .await
            .unwrap();

        

        // Here you'd configure pipelines, etc. asynchronously
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let window_attributes = if self.is_web {
        #[cfg(target_arch = "wasm32")]
        {
            WindowAttributes {
                title: String::from("3D Web Engine"),
                ..Default::default()
            }
            .with_canvas(Some("#wasm-canvas"))
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            panic!("is_web=true but not compiling for wasm32 target");
        }
        } else {
            WindowAttributes {
                title: String::from("3D Web Engine"),
                ..Default::default()
            }
        };
    
        let window = event_loop.create_window(window_attributes).unwrap();
        let size = window.inner_size();

        //
        self.window = Some(window.clone());
        let ready_flag = self.engine_ready.clone();
        let is_web = self.is_web;
        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                App::init_engine(window, is_web).await;
                ready_flag.store(true, Ordering::SeqCst); // signal ready
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            std::thread::spawn(move || {
                pollster::block_on(async {
                    App::init_engine(window, is_web).await;
                    ready_flag.store(true, Ordering::SeqCst); // signal ready
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

                    if self.engine_ready.load(Ordering::SeqCst) {
                        //Here you'd actually render with wgpu
                        // (currently we just clear and present)
                        // For now, nothing to draw
                    }

                    if let Some(win) = &self.window {
                        win.request_redraw();
                    }
                }
                //self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    // let event_loop = EventLoop::new().unwrap();

    // // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // // dispatched any events. This is ideal for games and similar applications.
    // event_loop.set_control_flow(ControlFlow::Poll);

    // // ControlFlow::Wait pauses the event loop if no events are available to process.
    // // This is ideal for non-game applications that only update in response to user
    // // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    // event_loop.set_control_flow(ControlFlow::Wait);

    // let mut app = App::default();
    // event_loop.run_app(&mut app);
    pollster::block_on(run(false));
}

