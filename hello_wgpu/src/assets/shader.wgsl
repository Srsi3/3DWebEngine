struct Camera {
    viewProj : mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> uCamera : Camera;

struct VsIn {
    @location(0) position : vec3<f32>,
    @location(1) color    : vec4<f32>,
};

struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) color     : vec4<f32>,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out : VsOut;
    let world_pos = vec4<f32>(in.position, 1.0);   // no model matrix yet
    out.pos   = uCamera.viewProj * world_pos;      // apply camera
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
