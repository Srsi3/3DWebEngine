// src/culling.rs
use cgmath::{Matrix4, Vector3, InnerSpace};

#[derive(Copy, Clone, Debug)]
pub struct Plane {
    pub n: Vector3<f32>, // normalized normal
    pub d: f32,          // distance term
}

#[derive(Copy, Clone, Debug)]
pub struct Frustum {
    pub planes: [Plane; 6], // left, right, bottom, top, near, far
}

fn normalize_plane(mut p: Plane) -> Plane {
    let len = (p.n.x * p.n.x + p.n.y * p.n.y + p.n.z * p.n.z).sqrt().max(1e-6);
    p.n.x /= len; p.n.y /= len; p.n.z /= len;
    p.d    /= len;
    p
}

/// Extracts planes from a column-major CGMath Matrix4 (VP = P * V).
/// We build ROW vectors explicitly:
/// row0 = [ m.x.x, m.y.x, m.z.x, m.w.x ], etc.
pub fn frustum_from_vp(vp: &Matrix4<f32>) -> Frustum {
    let m = vp;
    let r0 = [ m.x.x, m.y.x, m.z.x, m.w.x ];
    let r1 = [ m.x.y, m.y.y, m.z.y, m.w.y ];
    let r2 = [ m.x.z, m.y.z, m.z.z, m.w.z ];
    let r3 = [ m.x.w, m.y.w, m.z.w, m.w.w ];

    // Combine rows per Gribb/Hartmann
    let planes = [
        // Left:  r3 + r0
        Plane { n: Vector3::new(r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2]), d: r3[3] + r0[3] },
        // Right: r3 - r0
        Plane { n: Vector3::new(r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2]), d: r3[3] - r0[3] },
        // Bottom:r3 + r1
        Plane { n: Vector3::new(r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2]), d: r3[3] + r1[3] },
        // Top:   r3 - r1
        Plane { n: Vector3::new(r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2]), d: r3[3] - r1[3] },
        // Near:  r3 + r2
        Plane { n: Vector3::new(r3[0] + r2[0], r3[1] + r2[1], r3[2] + r2[2]), d: r3[3] + r2[3] },
        // Far:   r3 - r2
        Plane { n: Vector3::new(r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2]), d: r3[3] - r2[3] },
    ].map(normalize_plane);

    Frustum { planes }
}

/// AABB vs frustum test (positive-vertex radius trick).
/// center = AABB center; half = half-extents. Returns true if intersects.
pub fn aabb_intersects_frustum(center: Vector3<f32>, half: Vector3<f32>, fr: &Frustum) -> bool {
    for p in &fr.planes {
        // Project AABB onto plane normal to get the support radius
        let r = half.x * p.n.x.abs() + half.y * p.n.y.abs() + half.z * p.n.z.abs();
        // Signed distance from center to plane
        let s = p.n.dot(center) + p.d;
        if s < -r {
            return false; // completely outside this plane
        }
    }
    true
}
