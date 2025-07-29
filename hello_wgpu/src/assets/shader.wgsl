struct Camera {
    view_proj : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> uCamera : Camera;

struct VSIn {
    @location(0) position : vec3<f32>,
    @location(1) color    : vec4<f32>,

    // per-instance
    @location(2) inst_pos   : vec4<f32>, // pos.xyz
    @location(3) inst_scale : vec4<f32>, // scale.xyz
};

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(in: VSIn) -> VSOut {
    // Scale around origin, then translate by center
    let world_pos = in.inst_pos.xyz + in.inst_scale.xyz * in.position;
    var out : VSOut;
    out.pos   = uCamera.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return in.color;
}
