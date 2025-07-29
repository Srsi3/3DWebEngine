struct Camera {
    viewProj : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uCamera : Camera;

struct VsIn {
    @location(0) position : vec3<f32>,
    @location(1) color    : vec4<f32>,
    // Per-instance model matrix (4x vec4)
    @location(2) i_m0     : vec4<f32>,
    @location(3) i_m1     : vec4<f32>,
    @location(4) i_m2     : vec4<f32>,
    @location(5) i_m3     : vec4<f32>,
};

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) color     : vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out : VsOut;

    let model = mat4x4<f32>(in.i_m0, in.i_m1, in.i_m2, in.i_m3);
    let world_pos = model * vec4<f32>(in.position, 1.0);

    out.pos   = uCamera.viewProj * world_pos;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
