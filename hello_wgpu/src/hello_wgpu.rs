//! winit glue: toroidal wrap + floating-origin, palette, live mutations,
//! per-archetype batching (low-rise demo) and debug controls.

use std::sync::{Arc, atomic::{AtomicBool, Ordering}, Mutex};

use cgmath::{EuclideanSpace, InnerSpace, Matrix4, Vector3};
use instant::Instant;
use log::{info, warn, error};
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::PhysicalKey,
    window::{Window, WindowAttributes, WindowId},
};

#[cfg(target_arch = "wasm32")]
use {
    wasm_bindgen::prelude::*,
    wasm_bindgen::JsCast,
    web_sys::HtmlCanvasElement,
    winit::platform::web::WindowAttributesExtWebSys,
};

use crate::{
    assets::{AssetLibrary, BuildingCategory},
    camera,
    chunking::{ChunkManager, ViewerId},
    culling,
    designer_ml::{RuleDesigner, CityDesigner},
    mesh,
    net_mutations,
    render::Engine,
    types::InstanceRaw,
};

// ───────────────────────── logging ─────────────────────────
fn init_logging(web: bool) {
    #[cfg(target_arch = "wasm32")]
    { let _ = console_log::init_with_level(log::Level::Info); }

    #[cfg(not(target_arch = "wasm32"))] {
        let env = env_logger::Env::default()
            .filter_or("RUST_LOG", "hello_wgpu=info,wgpu_core=warn,wgpu_hal=warn");
        let _ = env_logger::Builder::from_env(env).try_init();
    }
    info!("logging ready (web={})", web);
}

// ───────────────────────── public entry ─────────────────────
pub async fn run(is_web: bool) {
    init_logging(is_web);
    let el = EventLoop::new().expect("EL");
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(is_web);
    if let Err(e) = el.run_app(&mut app) {
        error!("event-loop error: {e:?}");
    }
}

// ───────────────────────── App struct ───────────────────────
struct App {
    // gfx
    is_web: bool,
    window: Option<Window>,
    surface: Option<wgpu::Surface<'static>>,
    adapter: Option<wgpu::Adapter>,
    engine:  Option<Engine>,

    // input & cam
    keyboard: camera::KeyboardInput,
    camera:   camera::Camera,
    last_cursor: Option<PhysicalPosition<f64>>,

    // timing
    last_frame: Instant,

    // async device create
    ready: Arc<AtomicBool>,
    gpu_slot: Arc<Mutex<Option<(wgpu::Device,wgpu::Queue)>>>,
    ad_slot:  Arc<Mutex<Option<wgpu::Adapter>>>,
    instance: Option<wgpu::Instance>,

    // world
    chunk_mgr: ChunkManager,
    designer:  RuleDesigner,
    viewer_id: ViewerId,
    world_origin: cgmath::Vector3<f64>,

    // ground inst
    ground_inst: InstanceRaw,

    // LOD / cull
    lod0:f32, lod1:f32, cull:f32,

    // misc
    debug: bool,
    dbg_last: Instant,
}

impl App {
    fn new(is_web: bool) -> Self {
        // generation parameters
        let params = crate::chunking::CityGenParams {
        lots_x:3, lots_z:3,
        lot_w:3.0, lot_d:3.0, lot_gap:0.4,
        road_w_minor:3.0, road_w_major:8.0, major_every:6,
        blocks_per_chunk_x:8, blocks_per_chunk_z:8,
        seed:0xA11CE_u64,
    };
    let bounds = (-4,4,-4,4);

    let chunk_mgr = ChunkManager::new(params.clone(), 3, bounds, true, "./city_chunks");

    let designer  = RuleDesigner { params };
        Self {
            is_web,
            window: None, surface: None, adapter: None, engine: None,
            keyboard: camera::KeyboardInput::new(),
            camera:   camera::Camera::new(),
            last_cursor: None,
            last_frame: Instant::now(),
            ready: Arc::new(AtomicBool::new(false)),
            gpu_slot: Arc::new(Mutex::new(None)),
            ad_slot:  Arc::new(Mutex::new(None)),
            instance: None,
            chunk_mgr: chunk_mgr,
            designer:  designer,
            viewer_id: 0,
            world_origin: cgmath::vec3(0.0,0.0,0.0),
            ground_inst: InstanceRaw {
                pos:[0.0,-0.05,0.0,0.0],
                scale:[1.0,1.0,1.0,0.0],
                misc:[2.0,0.0,0.0,0.0], // category=2 (landmark colour)
            },
            lod0:90.0, lod1:190.0, cull:380.0,
            debug:false, dbg_last:Instant::now(),
        }
    }

    // ------------ async device helper ------------
    async fn spawn_device(adapter: wgpu::Adapter,
                          slot: Arc<Mutex<Option<(wgpu::Device,wgpu::Queue)>>> ,
                          flag: Arc<AtomicBool>) {
        let (device,queue) = adapter.request_device(&wgpu::DeviceDescriptor::default()).await.unwrap();
        device.on_uncaptured_error(Box::new(|e| error!("WGPU uncaptured {e:?}")));
        { *slot.lock().unwrap() = Some((device,queue)); }
        flag.store(true,Ordering::SeqCst);
    }

    // ------------ engine creation ------------
    fn finalize(&mut self) {
        if self.engine.is_some() || !self.ready.load(Ordering::SeqCst) { return; }
        let (device,queue) = self.gpu_slot.lock().unwrap().take().unwrap();
        let surface = self.surface.take().unwrap();
        let adapter = if let Some(a)=&self.adapter { a.clone() }
                      else { self.ad_slot.lock().unwrap().take().unwrap() };
        let size = self.window.as_ref().unwrap().inner_size();
        self.engine = Some(Engine::new(device,queue,surface,&adapter,size));
    }

    // ------------ floating origin & torus wrap ------------
    const SHIFT_DIST: f32 = 500.0;
    fn maybe_float_origin(&mut self){
        let p=self.camera.position.to_vec();
        if p.magnitude() > Self::SHIFT_DIST { self.shift_world(p); }
    }
    fn shift_world(&mut self, off: Vector3<f32>){
        self.chunk_mgr.apply_shift(off);
        self.camera.position -= off;
        self.world_origin += cgmath::vec3(off.x as f64,0.0,off.z as f64);
    }
    fn maybe_wrap_torus(&mut self){
        let (sx,sz) = self.chunk_mgr.world_span();
        let hx=sx*0.5; let hz=sz*0.5;
        let mut off=Vector3::new(0.0,0.0,0.0);
        if self.camera.position.x >  hx { off.x =  sx; }
        if self.camera.position.x < -hx { off.x = -sx; }
        if self.camera.position.z >  hz { off.z =  sz; }
        if self.camera.position.z < -hz { off.z = -sz; }
        if off.x!=0.0||off.z!=0.0 { self.shift_world(off); }
    }
}

// ─────────────────── winit plumbing ────────────────────────
impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        // ---------- Window ----------
        let attrs = if self.is_web {
            #[cfg(target_arch="wasm32")]
            {
                let doc=web_sys::window().unwrap().document().unwrap();
                let cv = doc.get_element_by_id("wasm-canvas")
                            .expect("canvas").dyn_into::<HtmlCanvasElement>().unwrap();
                if cv.width()==0 || cv.height()==0 {
                    let w=cv.client_width() as u32; let h=cv.client_height() as u32;
                    cv.set_width(w); cv.set_height(h);
                }
                WindowAttributes::default().with_title("Techno-Medieval").with_canvas(Some(cv))
            }
            #[cfg(not(target_arch="wasm32"))] { unreachable!() }
        } else {
            WindowAttributes::default().with_title("Techno-Medieval")
        };
        let win = el.create_window(attrs).unwrap();
        self.window = Some(win);

        // ---------- Instance & Surface ----------
        let backends = if self.is_web { wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL }
                       else { wgpu::Backends::all() };
        let inst = wgpu::Instance::new(&wgpu::InstanceDescriptor{backends,..Default::default()});
        self.instance = Some(inst);
        let surf = unsafe{
            self.instance.as_ref().unwrap()
                .create_surface(self.window.as_ref().unwrap()).unwrap()
        };
        self.surface = Some(unsafe{ std::mem::transmute(surf) });

        let ready = self.ready.clone();
        let slot  = self.gpu_slot.clone();
        let adslot= self.ad_slot.clone();

        #[cfg(not(target_arch="wasm32"))] {
            let adapter = pollster::block_on(self.instance.as_ref().unwrap()
                .request_adapter(&wgpu::RequestAdapterOptions{
                    power_preference:wgpu::PowerPreference::HighPerformance,
                    compatible_surface:self.surface.as_ref(),
                    force_fallback_adapter:false,
                })).unwrap();
            self.adapter = Some(adapter.clone());
            std::thread::spawn(move||{
                pollster::block_on(Self::spawn_device(adapter,slot,ready));
            });
        }
        #[cfg(target_arch="wasm32")] {
            wasm_bindgen_futures::spawn_local({
                let inst=self.instance.as_ref().unwrap().clone();
                async move {
                    let adapter = inst.request_adapter(&wgpu::RequestAdapterOptions{
                        power_preference:wgpu::PowerPreference::HighPerformance,
                        compatible_surface:None, force_fallback_adapter:false,
                    }).await.unwrap();
                    { *adslot.lock().unwrap() = Some(adapter.clone()); }
                    Self::spawn_device(adapter,slot,ready).await;
                }
            });
        }
    }

    fn about_to_wait(&mut self, _:&ActiveEventLoop) {
        if let Some(w)=&self.window { w.request_redraw(); }
    }

    fn window_event(&mut self, el:&ActiveEventLoop, id:WindowId, ev:WindowEvent) {
        if Some(id)!=self.window.as_ref().map(|w|w.id()) { return; }

        match ev {
            WindowEvent::CloseRequested => el.exit(),

            WindowEvent::KeyboardInput{event,..} =>{
                if let PhysicalKey::Code(code)=event.physical_key {
                    match event.state {
                        ElementState::Pressed   => self.keyboard.key_press(code),
                        ElementState::Released  => self.keyboard.key_release(code),
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } =>{
                if let Some(prev)=self.last_cursor.replace(position) {
                    let dx=(position.x-prev.x) as f32;
                    let dy=(position.y-prev.y) as f32;
                    self.camera.process_mouse_delta(dx,dy,0.002);
                }
            }
            WindowEvent::Resized(sz) =>{
                if let Some(e)=self.engine.as_mut(){ e.resize(sz); }
            }
            WindowEvent::RedrawRequested =>{
                let now=Instant::now();
                let dt=now.duration_since(self.last_frame).as_secs_f32();
                if dt<0.016 { return; }
                self.last_frame=now;

                self.camera.update(dt,&self.keyboard);
                self.maybe_wrap_torus();
                self.maybe_float_origin();
                self.finalize();
                
                let mut v0_low_common = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v0_low_alt    = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v0_high       = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v0_land       = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v1_low_common = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v1_low_alt    = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v1_high       = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v1_land       = Vec::<InstanceRaw>::with_capacity(4096);
                let mut v2_bill       = Vec::<InstanceRaw>::with_capacity(4096);


                if let Some(e)=self.engine.as_mut() {
                    // -------- (immutable borrow of assets scoped) --------
                    {
                        let assets:&AssetLibrary = e.assets_ref();

                        // network mutate packets
                        net_mutations::poll_incoming(&mut self.chunk_mgr, assets);

                        // chunk ensure + local mutations
                        self.chunk_mgr.set_viewer(self.viewer_id, self.camera.position.x, self.camera.position.z);
                        self.chunk_mgr.ensure_for_viewers(&mut self.designer, assets);

                        //info!("loaded chunks = {}", self.chunk_mgr.loaded.len());

                        self.chunk_mgr.mutate_near(assets, 0.02, dt, 1, now.elapsed().as_nanos() as u64);
                        
                        // build instance buckets
                        let size=self.window.as_ref().unwrap().inner_size();
                        let aspect=size.width.max(1) as f32 / size.height.max(1) as f32;
                        let vp=self.camera.view_projection(aspect);
                        e.update_camera(&vp);

                        let fr=culling::frustum_from_vp(&vp);
                        let cam=self.camera.position.to_vec();

                        // buckets
                       

                        // alt low-rise archetype id (timber_house_b = id 1)
                        let alt_id:usize = 1;

                        for list in self.chunk_mgr.loaded.values() {
                            for b in list {
                                let dist=(b.center-cam).magnitude();
                                if dist>self.cull { continue; }

                                let base=assets.base_half(b.archetype_id as usize);
                                let half=Vector3::new(
                                    base.x*b.scale.x, base.y*b.scale.y, base.z*b.scale.z);
                                if !culling::aabb_intersects_frustum(b.center,half,&fr){continue;}

                                let cat=assets.category_of(b.archetype_id as usize);
                                let inst=InstanceRaw{
                                    pos:[b.center.x,b.center.y,b.center.z,0.0],
                                    scale:[b.scale.x,b.scale.y,b.scale.z,0.0],
                                    misc:[match cat{
                                        BuildingCategory::Lowrise =>0.0,
                                        BuildingCategory::Highrise=>1.0,
                                        BuildingCategory::Landmark=>2.0,
                                    }, b.archetype_id as f32,0.0,0.0],
                                };

                                if dist<=self.lod0 {
                                    match cat {
                                        BuildingCategory::Lowrise=>{
                                            if b.archetype_id as usize==alt_id {
                                                v0_low_alt.push(inst)
                                            } else { v0_low_common.push(inst) }
                                        }
                                        BuildingCategory::Highrise => v0_high.push(inst),
                                        BuildingCategory::Landmark => v0_land.push(inst),
                                    }
                                } else if dist<=self.lod1 {
                                    match cat {
                                        BuildingCategory::Lowrise=>{
                                            if b.archetype_id as usize==alt_id {
                                                v1_low_alt.push(inst)
                                            } else { v1_low_common.push(inst) }
                                        }
                                        BuildingCategory::Highrise => v1_high.push(inst),
                                        BuildingCategory::Landmark => v1_land.push(inst),
                                    }
                                } else {
                                    v2_bill.push(InstanceRaw{
                                        pos:[b.center.x,b.center.y,b.center.z,0.0],
                                        scale:[half.x.max(0.5), (half.y*2.0).max(0.5),1.0,0.0],
                                        misc:[1.0,0.0,0.0,0.0], // tint = high-rise colour for far billboard
                                    });
                                }
                            }
                        }

                        let assets: &AssetLibrary = e.assets_ref();
                    }

                    e.update_instances(
                        &v0_low_common,&v0_low_alt,&v0_high,&v0_land,
                        &v1_low_common,&v1_low_alt,&v1_high,&v1_land,
                        &v2_bill,&self.ground_inst,
                    );
                    if let Err(err)=e.render(){
                        match err {
                            wgpu::SurfaceError::Lost|wgpu::SurfaceError::Outdated=>{
                                e.resize(self.window.as_ref().unwrap().inner_size());
                            }
                            wgpu::SurfaceError::OutOfMemory=>{
                                error!("OOM"); el.exit();
                            }
                            _=>warn!("surface err {err:?}"),
                        }
                    }
                }
                if let Some(w)=&self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}
