// ── net_mutations.rs ───────────────────────────────────────
use std::net::{UdpSocket, SocketAddr};
use std::sync::OnceLock;                // ← add
use crate::chunking::{ChunkManager, RuntimePlacement};
use crate::assets::AssetLibrary;
use crate::designer_ml::hash2;

static BROADCAST_ADDR: &str = "239.20.20.20:17017";
static SOCK: OnceLock<UdpSocket> = OnceLock::new();   // ← replace static mut

pub fn poll_incoming(cm: &mut ChunkManager, assets: &AssetLibrary) {
    // Initialise on first call, then reuse
    let sock: &UdpSocket = SOCK.get_or_init(|| {
        let sock = UdpSocket::bind("0.0.0.0:0").expect("udp bind");
        sock.set_nonblocking(true).ok();
        sock.join_multicast_v4(
            &"239.20.20.20".parse().unwrap(),
            &"0.0.0.0".parse().unwrap(),
        )
        .ok();
        sock
    });

    let mut buf = [0u8; 12];
    while let Ok((n, _src)) = sock.recv_from(&mut buf) {
        if n == 12 {
            let key = i32::from_le_bytes(buf[0..4].try_into().unwrap());
            let idx = u32::from_le_bytes(buf[4..8].try_into().unwrap()) as usize;
            let aid = u16::from_le_bytes(buf[8..10].try_into().unwrap());
            let sc  = u16::from_le_bytes(buf[10..12].try_into().unwrap());

            let cz = key & 0xFFFF;
            let cx = key >> 16;
            if let Some(list) = cm.loaded.get_mut(&crate::chunking::ChunkKey(cx, cz)) {
                if idx < list.len() {
                    list[idx].archetype_id = aid;
                    let j = (sc as f32) / 65535.0 * 0.2 + 0.9;
                    list[idx].scale.x *= j;
                    list[idx].scale.y *= j;
                    list[idx].scale.z *= j;
                    let base = assets.base_half(aid as usize);
                    list[idx].center.y = base.y * list[idx].scale.y;
                }
            }
        }
    }
}
