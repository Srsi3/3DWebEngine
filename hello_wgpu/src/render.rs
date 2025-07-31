use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use log::info;
use wgpu::util::DeviceExt;

use crate::assets::{AssetLibrary, CategoryMesh, BuildingCategory};
use crate::mesh;
use crate::types::{CameraUniform, InstanceRaw, instance_buffer_layout};

// ───────────────────────────────── Palette ────────────────────────────────
const PALETTE_BYTES: u64 = 256;
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GpuPalette {
    low:  [f32; 3],
    high: [f32; 3],
    land: [f32; 3],
}
impl Default for GpuPalette {
    fn default() -> Self { Self {
        low:  [0.55, 0.40, 0.30],
        high: [0.25, 0.28, 0.30],
        land: [0.60, 0.48, 0.10],
    }}
}

// helper
fn ensure_buf(device: &wgpu::Device, buf: &mut wgpu::Buffer, needed: usize, label: &str) {
    let elem = std::mem::size_of::<InstanceRaw>() as u64;
    let req_bytes = (needed.max(1) as u64) * elem;
    if req_bytes <= buf.size() { return; }
    let new_sz = (req_bytes as f32 * 1.5).ceil() as u64;
    *buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label), size: new_sz,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
}

// ───────────────────────────────── Engine ────────────────────────────────
pub struct Engine {
    pub device: wgpu::Device,
    pub queue:  wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config:  wgpu::SurfaceConfiguration,

    shader: wgpu::ShaderModule,
    pipeline_layout: wgpu::PipelineLayout,
    render_pipeline: wgpu::RenderPipeline,

    // depth
    depth_format: wgpu::TextureFormat,
    depth_view:   wgpu::TextureView,

    // camera
    camera_bgl: wgpu::BindGroupLayout,
    camera_bg:  wgpu::BindGroup,
    camera_buf: wgpu::Buffer,

    // palette
    palette_bgl: wgpu::BindGroupLayout,
    palette_bg:  wgpu::BindGroup,
    palette_buf: wgpu::Buffer,

    // asset library (meshes + archetypes)
    pub assets: AssetLibrary,

    // instance buffers (category level)
    buf_ground: wgpu::Buffer,
    buf_l0_low_common:  wgpu::Buffer,
    buf_l0_low_alt:     wgpu::Buffer,
    buf_l0_high: wgpu::Buffer,
    buf_l0_land: wgpu::Buffer,
    buf_l1_low_common:  wgpu::Buffer,
    buf_l1_low_alt:     wgpu::Buffer,
    buf_l1_high: wgpu::Buffer,
    buf_l1_land: wgpu::Buffer,
    buf_l2_bill: wgpu::Buffer,

    // draw counts
    cnt_ground: u32,
    cnt_l0_low_common: u32,
    cnt_l0_low_alt:    u32,
    cnt_l0_high: u32,
    cnt_l0_land: u32,
    cnt_l1_low_common: u32,
    cnt_l1_low_alt:    u32,
    cnt_l1_high: u32,
    cnt_l1_land: u32,
    cnt_l2_bill: u32,
}

impl Engine {
    pub fn assets_ref(&self) -> &AssetLibrary { &self.assets }

    pub fn new(
        device: wgpu::Device,
        queue:  wgpu::Queue,
        mut surface: wgpu::Surface<'static>,
        adapter: &wgpu::Adapter,
        size: winit::dpi::PhysicalSize<u32>,
    ) -> Self {
        // Surface config
        let caps = surface.get_capabilities(adapter);
        let format = caps.formats[0];
        let alpha = if caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::Opaque) {
            wgpu::CompositeAlphaMode::Opaque
        } else { caps.alpha_modes[0] };
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format, width: size.width, height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: alpha,
            view_formats: vec![],
            desired_maximum_frame_latency: 0,
        };
        surface.configure(&device, &config);

        // Depth
        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let depth_view = {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("depth"), size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                mip_level_count:1, sample_count:1, dimension: wgpu::TextureDimension::D2,
                format: depth_format, usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats:&[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
        });

        // Camera group
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label:Some("camera bgl"),
            entries:&[wgpu::BindGroupLayoutEntry{
                binding:0,
                visibility:wgpu::ShaderStages::VERTEX,
                ty:wgpu::BindingType::Buffer{
                    ty:wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset:false,
                    min_binding_size:wgpu::BufferSize::new(64),
                },
                count:None,
            }],
        });
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor{
            label:Some("camera buf"),
            size:256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor{
            label:Some("camera bg"),
            layout:&camera_bgl,
            entries:&[wgpu::BindGroupEntry{binding:0,resource:camera_buf.as_entire_binding()}],
        });

        // Palette group
        let palette_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
            label: Some("palette bgl"),
            entries: &[wgpu::BindGroupLayoutEntry{
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(PALETTE_BYTES),  // 256
                },
                count: None,
            }],
        });

        let palette_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("palette buf"),
            size: PALETTE_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        queue.write_buffer(&palette_buf, 0, bytemuck::bytes_of(&GpuPalette::default()));

        let palette_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("palette bg"),
            layout: &palette_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: palette_buf.as_entire_binding(),
            }],
        });

        // Pipeline
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
            label:Some("pipe layout"),
            bind_group_layouts:&[&camera_bgl,&palette_bgl],
            push_constant_ranges:&[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
            label:Some("pipe"),
            layout:Some(&pipeline_layout),
            vertex: wgpu::VertexState{
                module:&shader,
                entry_point:Some("vs_main"),
                compilation_options:Default::default(),
                buffers:&[mesh::Vertex::layout(), instance_buffer_layout()],
            },
            fragment:Some(wgpu::FragmentState{
                module:&shader,
                entry_point:Some("fs_main"),
                compilation_options:Default::default(),
                targets:&[Some(wgpu::ColorTargetState{
                    format:config.format,
                    blend:Some(wgpu::BlendState::REPLACE),
                    write_mask:wgpu::ColorWrites::ALL,
                })],
            }),
            primitive:wgpu::PrimitiveState::default(),
            depth_stencil:Some(wgpu::DepthStencilState{
                format:depth_format,
                depth_write_enabled:true,
                depth_compare:wgpu::CompareFunction::Less,
                stencil:wgpu::StencilState::default(),
                bias:wgpu::DepthBiasState::default(),
            }),
            multisample:wgpu::MultisampleState::default(),
            multiview:None,
            cache:None,
        });

        // Assets
        let assets = AssetLibrary::new(&device);

        // Tiny helpers
        let mk = |lbl:&str| device.create_buffer(&wgpu::BufferDescriptor{
            label:Some(lbl),
            size: std::mem::size_of::<InstanceRaw>() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation:false,
        });
        let buf_ground = mk("inst ground");

        let buf_l0_low_common = mk("l0 low common");
        let buf_l0_low_alt    = mk("l0 low alt");
        let buf_l0_high   = mk("l0 high");
        let buf_l0_land   = mk("l0 land");
        let buf_l1_low_common = mk("l1 low common");
        let buf_l1_low_alt    = mk("l1 low alt");
        let buf_l1_high   = mk("l1 high");
        let buf_l1_land   = mk("l1 land");
        let buf_l2_bill   = mk("l2 bill");

        Self {
            device, queue, surface, config,
            shader, pipeline_layout, render_pipeline,
            depth_format, depth_view,
            camera_bgl, camera_bg, camera_buf,
            palette_bgl, palette_bg, palette_buf,
            assets,
            buf_ground,
            buf_l0_low_common, buf_l0_low_alt, buf_l0_high, buf_l0_land,
            buf_l1_low_common, buf_l1_low_alt, buf_l1_high, buf_l1_land,
            buf_l2_bill,
            cnt_ground:0,
            cnt_l0_low_common:0, cnt_l0_low_alt:0, cnt_l0_high:0, cnt_l0_land:0,
            cnt_l1_low_common:0, cnt_l1_low_alt:0, cnt_l1_high:0, cnt_l1_land:0,
            cnt_l2_bill:0,
        }
    }

    // ---------- window resize ----------
    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width==0 || new_size.height==0 { return; }
        self.config.width=new_size.width; self.config.height=new_size.height;
        self.surface.configure(&self.device,&self.config);
        let tex=self.device.create_texture(&wgpu::TextureDescriptor{
            label:Some("depth"),
            size: wgpu::Extent3d{width:new_size.width,height:new_size.height,depth_or_array_layers:1},
            mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
            format:self.depth_format,usage:wgpu::TextureUsages::RENDER_ATTACHMENT,view_formats:&[],
        });
        self.depth_view=tex.create_view(&wgpu::TextureViewDescriptor::default());
    }

    // ---------- camera ----------
    pub fn update_camera(&self, vp:&cgmath::Matrix4<f32>) {
        let data = CameraUniform{ view_proj:[
            [vp.x.x,vp.x.y,vp.x.z,vp.x.w],
            [vp.y.x,vp.y.y,vp.y.z,vp.y.w],
            [vp.z.x,vp.z.y,vp.z.z,vp.z.w],
            [vp.w.x,vp.w.y,vp.w.z,vp.w.w],
        ]};
        self.queue.write_buffer(&self.camera_buf,0,bytemuck::bytes_of(&data));
    }

    // ---------- instances ----------
    /// Call once per frame after culling.
    pub fn update_instances(
        &mut self,
        v0_low_common:&[InstanceRaw], v0_low_alt:&[InstanceRaw],
        v0_high:&[InstanceRaw], v0_land:&[InstanceRaw],
        v1_low_common:&[InstanceRaw], v1_low_alt:&[InstanceRaw],
        v1_high:&[InstanceRaw], v1_land:&[InstanceRaw],
        v2_bill:&[InstanceRaw],
        ground:&InstanceRaw,
    ){
        ensure_buf(&self.device,&mut self.buf_ground,1,"ground buf");
        ensure_buf(&self.device,&mut self.buf_l0_low_common,v0_low_common.len(),"l0 low com");
        ensure_buf(&self.device,&mut self.buf_l0_low_alt,v0_low_alt.len(),"l0 low alt");
        ensure_buf(&self.device,&mut self.buf_l0_high,v0_high.len(),"l0 high");
        ensure_buf(&self.device,&mut self.buf_l0_land,v0_land.len(),"l0 land");
        ensure_buf(&self.device,&mut self.buf_l1_low_common,v1_low_common.len(),"l1 low com");
        ensure_buf(&self.device,&mut self.buf_l1_low_alt,v1_low_alt.len(),"l1 low alt");
        ensure_buf(&self.device,&mut self.buf_l1_high,v1_high.len(),"l1 high");
        ensure_buf(&self.device,&mut self.buf_l1_land,v1_land.len(),"l1 land");
        ensure_buf(&self.device,&mut self.buf_l2_bill,v2_bill.len(),"l2 bill");

        self.queue.write_buffer(&self.buf_ground,0,bytemuck::bytes_of(ground));
        if !v0_low_common.is_empty(){ self.queue.write_buffer(&self.buf_l0_low_common,0,bytemuck::cast_slice(v0_low_common)); }
        if !v0_low_alt.is_empty()   { self.queue.write_buffer(&self.buf_l0_low_alt,   0,bytemuck::cast_slice(v0_low_alt)); }
        if !v0_high.is_empty()      { self.queue.write_buffer(&self.buf_l0_high,      0,bytemuck::cast_slice(v0_high)); }
        if !v0_land.is_empty()      { self.queue.write_buffer(&self.buf_l0_land,      0,bytemuck::cast_slice(v0_land)); }
        if !v1_low_common.is_empty(){ self.queue.write_buffer(&self.buf_l1_low_common,0,bytemuck::cast_slice(v1_low_common)); }
        if !v1_low_alt.is_empty()   { self.queue.write_buffer(&self.buf_l1_low_alt,   0,bytemuck::cast_slice(v1_low_alt)); }
        if !v1_high.is_empty()      { self.queue.write_buffer(&self.buf_l1_high,      0,bytemuck::cast_slice(v1_high)); }
        if !v1_land.is_empty()      { self.queue.write_buffer(&self.buf_l1_land,      0,bytemuck::cast_slice(v1_land)); }
        if !v2_bill.is_empty()      { self.queue.write_buffer(&self.buf_l2_bill,      0,bytemuck::cast_slice(v2_bill)); }

        self.cnt_ground = 1;
        self.cnt_l0_low_common = v0_low_common.len() as u32;
        self.cnt_l0_low_alt    = v0_low_alt.len()  as u32;
        self.cnt_l0_high       = v0_high.len()     as u32;
        self.cnt_l0_land       = v0_land.len()     as u32;
        self.cnt_l1_low_common = v1_low_common.len() as u32;
        self.cnt_l1_low_alt    = v1_low_alt.len()  as u32;
        self.cnt_l1_high       = v1_high.len()     as u32;
        self.cnt_l1_land       = v1_land.len()     as u32;
        self.cnt_l2_bill       = v2_bill.len()     as u32;

        info!("cnt0={} / cnt1={} / cnt2={}", self.cnt_l0_low_common, self.cnt_l1_low_common, self.cnt_l2_bill);
    }

    // ---------- draw ----------
    pub fn render(&mut self)->Result<(),wgpu::SurfaceError>{
        let frame=self.surface.get_current_texture()?;
        let view=frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder=self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{label:Some("enc")});

        {
            let mut rpass=encoder.begin_render_pass(&wgpu::RenderPassDescriptor{
                label:Some("main pass"),
                color_attachments:&[Some(wgpu::RenderPassColorAttachment{
                    view:&view,depth_slice:None,resolve_target:None,
                    ops:wgpu::Operations{load:wgpu::LoadOp::Clear(wgpu::Color{r:0.06,g:0.06,b:0.08,a:1.0}),store:wgpu::StoreOp::Store},
                })],
                depth_stencil_attachment:Some(wgpu::RenderPassDepthStencilAttachment{
                    view:&self.depth_view,
                    depth_ops:Some(wgpu::Operations{load:wgpu::LoadOp::Clear(1.0),store:wgpu::StoreOp::Store}),
                    stencil_ops:None,
                }),
                timestamp_writes:None, occlusion_query_set:None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0,&self.camera_bg,&[]);
            rpass.set_bind_group(1,&self.palette_bg,&[]);

            // Ground
            rpass.set_vertex_buffer(0,self.assets.mesh_ground.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.assets.mesh_ground.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(1,self.buf_ground.slice(..));
            rpass.draw_indexed(0..self.assets.mesh_ground.index_count,0,0..self.cnt_ground);

            // LOD0 lowrise: common + alt
            if self.cnt_l0_low_common>0 {
                rpass.set_vertex_buffer(0,self.assets.mesh_lowrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_lowrise.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l0_low_common.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_lowrise.index_count,0,0..self.cnt_l0_low_common);
            }
            if self.cnt_l0_low_alt>0 {
                let alt_mesh = self.assets.mesh_of(1/*timber_house_b*/).unwrap(); // assumes id=1
                rpass.set_vertex_buffer(0,alt_mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(alt_mesh.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l0_low_alt.slice(..));
                rpass.draw_indexed(0..alt_mesh.index_count,0,0..self.cnt_l0_low_alt);
            }
            // LOD0 highrise & landmark
            if self.cnt_l0_high>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_highrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_highrise.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l0_high.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_highrise.index_count,0,0..self.cnt_l0_high);
            }
            if self.cnt_l0_land>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_landmark.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_landmark.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l0_land.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_landmark.index_count,0,0..self.cnt_l0_land);
            }

            // LOD1 batches
            if self.cnt_l1_low_common>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_lowrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_lowrise.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l1_low_common.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_lowrise.index_count,0,0..self.cnt_l1_low_common);
            }
            if self.cnt_l1_low_alt>0{
                let alt_mesh=self.assets.mesh_of(1).unwrap();
                rpass.set_vertex_buffer(0,alt_mesh.vertex_buffer.slice(..));
                rpass.set_index_buffer(alt_mesh.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l1_low_alt.slice(..));
                rpass.draw_indexed(0..alt_mesh.index_count,0,0..self.cnt_l1_low_alt);
            }
            if self.cnt_l1_high>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_highrise.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_highrise.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l1_high.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_highrise.index_count,0,0..self.cnt_l1_high);
            }
            if self.cnt_l1_land>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_landmark.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_landmark.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l1_land.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_landmark.index_count,0,0..self.cnt_l1_land);
            }

            // LOD2 billboards
            if self.cnt_l2_bill>0{
                rpass.set_vertex_buffer(0,self.assets.mesh_billboard.vertex_buffer.slice(..));
                rpass.set_index_buffer(self.assets.mesh_billboard.index_buffer.slice(..),wgpu::IndexFormat::Uint16);
                rpass.set_vertex_buffer(1,self.buf_l2_bill.slice(..));
                rpass.draw_indexed(0..self.assets.mesh_billboard.index_count,0,0..self.cnt_l2_bill);
            }
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
