use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;

#[cfg(target_arch = "wasm32")]
use winit::platform::web::WindowAttributesExtWeb;


pub async fn run(is_web: bool) {
    let event_loop = EventLoop::new().unwrap();
    let mut app = App::new(is_web);
    event_loop.set_control_flow(ControlFlow::Poll);
    event_loop.run_app(&mut app).unwrap();
}

struct App {
    is_web: bool,
    window: Option<Window>,
}

impl App {
    fn new(is_web: bool) -> Self {
        Self { is_web, window: None }
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
        pollster::block_on(async {
            let backends = if self.is_web {
                wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL
            } else {
                wgpu::Backends::all()
            };
            let instance  = wgpu::Instance::new(&wgpu::InstanceDescriptor { backends, ..Default::default() });
            let surface   = unsafe { instance.create_surface(&window).unwrap() };
            let adapter   = instance.request_adapter(&wgpu::RequestAdapterOptions{
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: Some(&surface),
                    force_fallback_adapter: false,
                }).await.expect("No compatible adapter found (WebGPU+GL)");

            // ----- 4.3 acquire device/queue with default WebGPU limits -----
            let (_device, _queue) = adapter.request_device(&Default::default()).await.unwrap();
            });

        // original
        self.window = Some(window);
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, id: WindowId, event: WindowEvent) {
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
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => (),
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();

    // ControlFlow::Poll continuously runs the event loop, even if the OS hasn't
    // dispatched any events. This is ideal for games and similar applications.
    event_loop.set_control_flow(ControlFlow::Poll);

    // ControlFlow::Wait pauses the event loop if no events are available to process.
    // This is ideal for non-game applications that only update in response to user
    // input, and uses significantly less power/CPU time than ControlFlow::Poll.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::default();
    event_loop.run_app(&mut app);
}

