use wgpu::util::DeviceExt;
use crate::mesh;
use crate::types::{CameraUniform, InstanceRaw, instance_buffer_layout, mat4_to_array};

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

    // city meshes
    pub mesh_lowrise:   mesh::Mesh,
    pub mesh_highrise:  mesh::Mesh,
    pub mesh_pyramid:   mesh::Mesh,
    pub mesh_billboard: mesh::Mesh,
    pub mesh_ground:    mesh::Mesh,

    // instance buffers (auto-growing)
    ground_instbuf: wgpu::Buffer,
    instbuf_lod0_lowrise:  wgpu::Buffer,
    instbuf_lod0_highrise: wgpu::Buffer,
    instbuf_lod0_pyramid:  wgpu::Buffer,
    instbuf_lod1_lowrise:  wgpu::Buffer,
    instbuf_lod1_highrise: wgpu::Buffer,
    instbuf_lod1_pyramid:  wgpu::Buffer,
    instbuf_lod2_billboard: wgpu::Buffer,

    // draw counts (updated by update_instances)
    pub cnt_lod0_lowrise:  u32,
    pub cnt_lod0_highrise: u32,
    pub cnt_lod0_pyramid:  u32,
    pub cnt_lod1_lowrise:  u32,
    pub cnt_lod1_highrise: u32,
    pub cnt_lod1_pyramid:  u32,
    pub cnt_lod2_billboard: u32,
}

// ---- helper: grow buffer capacity without borrowing `self` ----
fn ensure_buf_capacity(
    device: &wgpu::Device,
    buf: &mut wgpu::Buffer,
    needed_instances: usize,
    label: &str,
) {
    let needed_bytes = (needed_instances.max(1) * std::mem::size_of::<InstanceRaw>()) as u64;
    if needed_bytes <= buf.size() {
        return;
    }
    // grow ~1.5x to reduce realloc frequency
    let new_bytes = (needed_bytes as f32 * 1.5).ceil() as u64;
    let newb = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label} (grown)")),
        size: new_bytes,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    *buf = newb;
}

impl Engine {
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
            format,
            width: size.width,
            height: size.height,
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
                label: Some("Depth"),
                size: wgpu::Extent3d { width: size.width, height: size.height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: depth_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };

        // Shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Main Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("assets/shader.wgsl").into()),
        });

        // Camera
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
            size: 256,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera BG"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() }],
        });

        // Pipeline
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
                buffers: &[mesh::Vertex::layout(), instance_buffer_layout()],
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

        // Meshes
        let city = mesh::create_city_meshes(&device);

        // Instance buffers (start small; will grow automatically)
        let mk = |label| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<InstanceRaw>() as u64, // 1 instance to start
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_instbuf      = mk("ground_instbuf");
        let instbuf_lod0_low    = mk("instbuf_lod0_lowrise");
        let instbuf_lod0_high   = mk("instbuf_lod0_highrise");
        let instbuf_lod0_pyr    = mk("instbuf_lod0_pyramid");
        let instbuf_lod1_low    = mk("instbuf_lod1_lowrise");
        let instbuf_lod1_high   = mk("instbuf_lod1_highrise");
        let instbuf_lod1_pyr    = mk("instbuf_lod1_pyramid");
        let instbuf_lod2_bill   = mk("instbuf_lod2_billboard");

        Self {
            device, queue, surface, config,
            shader, pipeline_layout, render_pipeline,
            depth_format, depth_view,
            camera_bgl, camera_bg, camera_buf,
            mesh_lowrise: city.lowrise,
            mesh_highrise: city.highrise,
            mesh_pyramid:  city.pyramid,
            mesh_billboard: city.billboard,
            mesh_ground:    city.ground,
            ground_instbuf,
            instbuf_lod0_lowrise:  instbuf_lod0_low,
            instbuf_lod0_highrise: instbuf_lod0_high,
            instbuf_lod0_pyramid:  instbuf_lod0_pyr,
            instbuf_lod1_lowrise:  instbuf_lod1_low,
            instbuf_lod1_highrise: instbuf_lod1_high,
            instbuf_lod1_pyramid:  instbuf_lod1_pyr,
            instbuf_lod2_billboard: instbuf_lod2_bill,
            cnt_lod0_lowrise: 0, cnt_lod0_highrise: 0, cnt_lod0_pyramid: 0,
            cnt_lod1_lowrise: 0, cnt_lod1_highrise: 0, cnt_lod1_pyramid: 0,
            cnt_lod2_billboard: 0,
        }
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 { return; }
        self.config.width  = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        // recreate depth
        let tex = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth"),
            size: wgpu::Extent3d { width: new_size.width, height: new_size.height, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.depth_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    }

    pub fn update_camera(&self, vp: &cgmath::Matrix4<f32>) {
        let data = CameraUniform { view_proj: mat4_to_array(vp) };
        self.queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&data));
    }

    pub fn update_instances(
        &mut self,
        v0_low: &[InstanceRaw], v0_high: &[InstanceRaw], v0_pyr: &[InstanceRaw],
        v1_low: &[InstanceRaw], v1_high: &[InstanceRaw], v1_pyr: &[InstanceRaw],
        v2_bill: &[InstanceRaw],
        ground: &InstanceRaw,
    ) {
        // ---- grow buffers if needed (NO &mut self receiver involved) ----
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod0_lowrise,  v0_low.len(),  "instbuf_lod0_lowrise");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod0_highrise, v0_high.len(), "instbuf_lod0_highrise");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod0_pyramid,  v0_pyr.len(),  "instbuf_lod0_pyramid");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod1_lowrise,  v1_low.len(),  "instbuf_lod1_lowrise");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod1_highrise, v1_high.len(), "instbuf_lod1_highrise");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod1_pyramid,  v1_pyr.len(),  "instbuf_lod1_pyramid");
        ensure_buf_capacity(&self.device, &mut self.instbuf_lod2_billboard, v2_bill.len(),"instbuf_lod2_billboard");

        // ---- upload data ----
        self.queue.write_buffer(&self.ground_instbuf, 0, bytemuck::bytes_of(ground));
        if !v0_low.is_empty()  { self.queue.write_buffer(&self.instbuf_lod0_lowrise,  0, bytemuck::cast_slice(v0_low)); }
        if !v0_high.is_empty() { self.queue.write_buffer(&self.instbuf_lod0_highrise, 0, bytemuck::cast_slice(v0_high)); }
        if !v0_pyr.is_empty()  { self.queue.write_buffer(&self.instbuf_lod0_pyramid,  0, bytemuck::cast_slice(v0_pyr)); }
        if !v1_low.is_empty()  { self.queue.write_buffer(&self.instbuf_lod1_lowrise,  0, bytemuck::cast_slice(v1_low)); }
        if !v1_high.is_empty() { self.queue.write_buffer(&self.instbuf_lod1_highrise, 0, bytemuck::cast_slice(v1_high)); }
        if !v1_pyr.is_empty()  { self.queue.write_buffer(&self.instbuf_lod1_pyramid,  0, bytemuck::cast_slice(v1_pyr)); }
        if !v2_bill.is_empty() { self.queue.write_buffer(&self.instbuf_lod2_billboard,0, bytemuck::cast_slice(v2_bill)); }

        self.cnt_lod0_lowrise   = v0_low.len()  as u32;
        self.cnt_lod0_highrise  = v0_high.len() as u32;
        self.cnt_lod0_pyramid   = v0_pyr.len()  as u32;
        self.cnt_lod1_lowrise   = v1_low.len()  as u32;
        self.cnt_lod1_highrise  = v1_high.len() as u32;
        self.cnt_lod1_pyramid   = v1_pyr.len()  as u32;
        self.cnt_lod2_billboard = v2_bill.len() as u32;
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Main Encoder") }
        );

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Main Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load:  wgpu::LoadOp::Clear(wgpu::Color { r: 0.06, g: 0.06, b: 0.08, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            rpass.set_pipeline(&self.render_pipeline);
            rpass.set_bind_group(0, &self.camera_bg, &[]);

            // ground (1 instance)
            rpass.set_vertex_buffer(0, self.mesh_ground.vertex_buffer.slice(..));
            rpass.set_index_buffer(self.mesh_ground.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            rpass.set_vertex_buffer(1, self.ground_instbuf.slice(..));
            rpass.draw_indexed(0..self.mesh_ground.index_count, 0, 0..1);

            // LOD0
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

            // LOD1
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

            // LOD2 billboards
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
}
