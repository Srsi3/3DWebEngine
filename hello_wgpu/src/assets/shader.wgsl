// ---------- shared structs ----------
struct Camera {
    view_proj : mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> CAMERA : Camera;

struct Palette {
    col_low   : vec3<f32>,
    col_high  : vec3<f32>,
    col_land  : vec3<f32>,
};
@group(1) @binding(0) var<uniform> PAL : Palette;

struct VSIn {
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    // instance
    @location(2) i_pos   : vec3<f32>,
    @location(3) i_scale : vec3<f32>,
    @location(4) i_misc  : vec3<f32>,   // .x = category (0,1,2)   .y = archetypeId (0..65535 in uint bits)
};

struct VSOut {
    @builtin(position) pos : vec4<f32>,
    @location(0) worldN    : vec3<f32>,
    @location(1) tint_idx  : f32,
    @location(2) arche_id  : f32,
};

@vertex
fn vs_main(v : VSIn) -> VSOut {
    let world_pos = v.i_pos + v.position * v.i_scale;
    let world_n   = v.normal;
    var out : VSOut;
    out.pos = CAMERA.view_proj * vec4<f32>(world_pos, 1.0);
    out.worldN  = world_n;
    out.tint_idx = v.i_misc.x;
    out.arche_id = v.i_misc.y;
    return out;
}

@fragment
fn fs_main(in : VSOut) -> @location(0) vec4<f32> {
    // pick tint
    var tint : vec3<f32>;
    if     (in.tint_idx < 0.5) { tint = PAL.col_low;  }
    else if(in.tint_idx < 1.5) { tint = PAL.col_high; }
    else                       { tint = PAL.col_land; }
    let light = clamp(dot(normalize(in.worldN), vec3<f32>(0.4,0.9,0.1)), 0.15, 1.0);
    return vec4<f32>(tint * light, 1.0);
}
