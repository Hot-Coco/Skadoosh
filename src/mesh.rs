//! Multi-agent mesh networking: LAN peer discovery via UDP broadcast, a
//! shared peer table, a tiny embedded HTTP server that accepts forwarded
//! calls, and routing of `forward_call` tool invocations to discovered peers.
//!
//! # Protocol
//!
//! Each node broadcasts a UDP datagram `skadoosh-mesh {name} {http_port}`
//! to the LAN broadcast address every [`ANNOUNCE_INTERVAL`] (10 s). Peers
//! hearing it record `{name} → ip:http_port` in their peer table. The UDP
//! discovery port and the HTTP port share [`Config::mesh_port`](crate::Config::mesh_port)
//! (default [`DEFAULT_MESH_PORT`], 9876) — UDP and TCP are distinct protocols,
//! so the two sockets never collide on the same port number.
//!
//! # Forwarding
//!
//! When the LLM invokes `forward_call` with a `target` peer name, the mesh
//! resolves the name to the peer's HTTP endpoint and POSTs the conversation
//! context to `http://{endpoint}/forward`, relaying the peer's text reply
//! back as the tool result. A call without a `target` falls back to the
//! `--forward-url` endpoint (see [`crate::forward`]).
//!
//! # Dependencies
//!
//! Discovery uses [`std::net::UdpSocket`] and the HTTP server uses
//! [`std::net::TcpListener`] with manual HTTP/1.1 parsing — no new crate
//! dependencies. The client side of a forward (POSTing to a peer) reuses the
//! crate's existing `reqwest` client.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, UdpSocket};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::error::{Result, SkadooshError};
use crate::llm::Message;

/// Magic prefix for mesh announcement datagrams:
/// `skadoosh-mesh {name} {http_port}`.
pub const MESH_ANNOUNCE_PREFIX: &str = "skadoosh-mesh";

/// Default UDP discovery + HTTP port for the mesh.
pub const DEFAULT_MESH_PORT: u16 = 9876;

/// How often a node broadcasts its announcement datagram.
pub const ANNOUNCE_INTERVAL: Duration = Duration::from_secs(10);

/// Peers not heard from within this window are pruned from the table.
pub const PEER_TTL: Duration = Duration::from_secs(30);

/// Read timeout for the discovery loop, so it can poll the shutdown flag.
const DISCOVERY_POLL: Duration = Duration::from_millis(500);

/// Accept-poll interval for the nonblocking HTTP listener.
const HTTP_POLL: Duration = Duration::from_millis(50);

/// Maximum request size the HTTP server will buffer (headers + body).
const HTTP_MAX_REQUEST: usize = 256 * 1024;

/// Builds the announcement datagram payload: `skadoosh-mesh {name} {http_port}`.
pub fn build_announce(name: &str, http_port: u16) -> String {
    format!("{MESH_ANNOUNCE_PREFIX} {name} {http_port}")
}

/// Parses an announcement datagram into `(name, http_port)`. Returns `None`
/// for anything that is not a well-formed `skadoosh-mesh {name} {port}` line.
///
/// ```
/// # use skadoosh::mesh::parse_announce;
/// assert_eq!(parse_announce("skadoosh-mesh alice 9876"), Some(("alice".to_string(), 9876)));
/// assert_eq!(parse_announce("hello"), None);
/// ```
pub fn parse_announce(msg: &str) -> Option<(String, u16)> {
    let mut parts = msg.split_whitespace();
    if parts.next()? != MESH_ANNOUNCE_PREFIX {
        return None;
    }
    let name = parts.next()?.to_string();
    let port: u16 = parts.next()?.parse().ok()?;
    Some((name, port))
}

/// A discovered peer: its agent name, HTTP endpoint (`host:port`), and when
/// it was last heard from.
///
/// `last_seen` (an [`Instant`]) is purely internal — used for expiry — so
/// `Peer` deliberately does not implement `Serialize`/`Deserialize`; callers
/// that need a serializable view use [`PeerTable::names`].
#[derive(Debug, Clone)]
pub struct Peer {
    /// The peer's agent name (unique key in the table).
    pub name: String,
    /// HTTP endpoint as `host:port` (e.g. `192.168.1.5:9876`).
    pub http_endpoint: String,
    /// When this peer was last heard from; used for expiry.
    pub last_seen: Instant,
}

/// The mutable, thread-safe registry of discovered peers, keyed by name.
///
/// Pure logic (record / lookup / prune / list) with no I/O, so it can be
/// unit-tested directly and driven by the discovery loop or by tests feeding
/// mock datagrams through [`handle_datagram`].
#[derive(Debug, Default)]
pub struct PeerTable {
    peers: HashMap<String, Peer>,
}

impl PeerTable {
    /// Creates an empty peer table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records (or refreshes) a peer: `name → http_endpoint`, stamped at
    /// `now`. Re-recording an existing name updates its endpoint and
    /// `last_seen`.
    pub fn record(&mut self, name: &str, http_endpoint: &str, now: Instant) {
        self.peers.insert(
            name.to_string(),
            Peer {
                name: name.to_string(),
                http_endpoint: http_endpoint.to_string(),
                last_seen: now,
            },
        );
    }

    /// Looks up a peer by name, returning a clone of its HTTP endpoint.
    pub fn lookup(&self, name: &str) -> Option<String> {
        self.peers.get(name).map(|p| p.http_endpoint.clone())
    }

    /// Removes peers whose `last_seen` is older than `ttl` relative to `now`.
    pub fn prune(&mut self, now: Instant, ttl: Duration) {
        self.peers
            .retain(|_, p| now.duration_since(p.last_seen) < ttl);
    }

    /// Returns a snapshot of all known peers.
    pub fn list(&self) -> Vec<Peer> {
        self.peers.values().cloned().collect()
    }

    /// Returns the sorted list of known peer names.
    pub fn names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.peers.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of known peers.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }
}

/// Feeds one received datagram into the table — the pure core of the
/// discovery loop, factored out for testing. A well-formed announce from a
/// peer other than `self_name` records `{name} → {src.ip}:{http_port}`;
/// self-announces and malformed lines are ignored.
pub fn handle_datagram(msg: &str, src: SocketAddr, self_name: &str, table: &mut PeerTable) {
    if let Some((name, http_port)) = parse_announce(msg) {
        if name != self_name {
            let endpoint = format!("{}:{}", src.ip(), http_port);
            table.record(&name, &endpoint, Instant::now());
        }
    }
}

/// One node in the mesh: owns its peer table and the background discovery +
/// HTTP-server threads. Cheap to drop: [`Drop`] signals shutdown and joins
/// the threads.
pub struct MeshNode {
    name: String,
    http_port: u16,
    http_addr: Option<SocketAddr>,
    udp_addr: Option<SocketAddr>,
    peers: Arc<Mutex<PeerTable>>,
    shutdown: Arc<AtomicBool>,
    handles: Vec<Option<JoinHandle<()>>>,
}

impl MeshNode {
    /// Starts a mesh node named `name`, binding the UDP discovery socket and
    /// the HTTP server to `port` (use `0` for an OS-assigned port). Binding
    /// failures degrade gracefully: the node is returned with the failing
    /// service disabled and a warning logged, so a mesh bind problem never
    /// aborts the whole agent.
    pub fn start(name: &str, port: u16) -> MeshNode {
        let peers = Arc::new(Mutex::new(PeerTable::new()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let mut handles: Vec<Option<JoinHandle<()>>> = Vec::new();
        let mut http_port = port;
        let mut http_addr: Option<SocketAddr> = None;
        let mut udp_addr: Option<SocketAddr> = None;

        // HTTP server first so the announce carries the real HTTP port.
        let listener = match TcpListener::bind(("0.0.0.0", port)) {
            Ok(l) => {
                http_addr = l.local_addr().ok();
                http_port = l.local_addr().map(|a| a.port()).unwrap_or(port);
                let _ = l.set_nonblocking(true);
                tracing::info!(agent = %name, http_addr = ?http_addr, "mesh HTTP server listening");
                Some(l)
            }
            Err(e) => {
                tracing::warn!(port, error = %e, "mesh: failed to bind HTTP listener; forward server disabled");
                None
            }
        };
        if let Some(listener) = listener {
            let peers_c = Arc::clone(&peers);
            let shut_c = Arc::clone(&shutdown);
            let name_c = name.to_string();
            handles.push(
                std::thread::Builder::new()
                    .name("skadoosh-mesh-http".to_string())
                    .spawn(move || http_loop(listener, name_c, peers_c, shut_c))
                    .ok(),
            );
        }

        // UDP discovery.
        match UdpSocket::bind(("0.0.0.0", port)) {
            Ok(sock) => {
                let _ = sock.set_broadcast(true);
                let udp_port = sock.local_addr().map(|a| a.port()).unwrap_or(port);
                udp_addr = sock.local_addr().ok();
                tracing::info!(agent = %name, udp_addr = ?udp_addr, "mesh UDP discovery listening");
                let peers_c = Arc::clone(&peers);
                let shut_c = Arc::clone(&shutdown);
                let name_c = name.to_string();
                let hp = http_port;
                handles.push(
                    std::thread::Builder::new()
                        .name("skadoosh-mesh-udp".to_string())
                        .spawn(move || discovery_loop(sock, name_c, hp, udp_port, peers_c, shut_c))
                        .ok(),
                );
            }
            Err(e) => {
                tracing::warn!(port, error = %e, "mesh: failed to bind UDP discovery socket; discovery disabled");
            }
        }

        MeshNode {
            name: name.to_string(),
            http_port,
            http_addr,
            udp_addr,
            peers,
            shutdown,
            handles,
        }
    }

    /// This node's agent name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The actual HTTP port the server bound (equals `mesh_port` unless `0`
    /// was requested, in which case it is OS-assigned).
    pub fn http_port(&self) -> u16 {
        self.http_port
    }

    /// The address the HTTP server is listening on, if it bound successfully.
    pub fn http_addr(&self) -> Option<SocketAddr> {
        self.http_addr
    }

    /// The address the UDP discovery socket is bound to, if it bound
    /// successfully. Send a `skadoosh-mesh {name} {port}` datagram here to
    /// simulate a peer announcement (used by tests).
    pub fn udp_addr(&self) -> Option<SocketAddr> {
        self.udp_addr
    }

    /// Shared handle to the peer table (for inspection / testing).
    pub fn peer_table(&self) -> &Arc<Mutex<PeerTable>> {
        &self.peers
    }

    /// Forwards the conversation context to the mesh peer named `target`:
    /// resolves the name to its HTTP endpoint, POSTs the full context to
    /// `http://{endpoint}/forward`, and relays the peer's text reply.
    ///
    /// Returns an error if the peer is unknown or the HTTP request fails.
    pub async fn forward_to_peer(
        &self,
        target: &str,
        history: &[Message],
        current_query: &str,
        reason: &str,
        summary: &str,
    ) -> Result<String> {
        let endpoint = self.peers.lock().unwrap().lookup(target).ok_or_else(|| {
            SkadooshError::Other(anyhow::anyhow!("mesh peer not found: {target}"))
        })?;
        let url = format!("http://{endpoint}/forward");
        tracing::info!(target = %target, url = %url, "mesh forwarding call to peer");

        let body = serde_json::json!({
            "target": target,
            "reason": reason,
            "summary": summary,
            "current_query": current_query,
            "history": history,
        });
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("mesh HTTP client: {e}")))?;
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("mesh forward request: {e}")))?;
        let status = resp.status();
        let text = resp
            .text()
            .await
            .map_err(|e| SkadooshError::Other(anyhow::anyhow!("mesh forward response: {e}")))?;
        if !status.is_success() {
            return Err(SkadooshError::Other(anyhow::anyhow!(
                "mesh peer {target} returned {status}: {}",
                text.chars().take(1024).collect::<String>()
            )));
        }
        Ok(text)
    }
}

impl Drop for MeshNode {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        for handle in self.handles.drain(..).flatten() {
            let _ = handle.join();
        }
    }
}

/// Background UDP discovery loop: receives peer announcements, records them,
/// and rebroadcasts this node's announcement every [`ANNOUNCE_INTERVAL`],
/// pruning stale peers as it goes. Exits when `shutdown` is set.
fn discovery_loop(
    sock: UdpSocket,
    name: String,
    http_port: u16,
    udp_port: u16,
    peers: Arc<Mutex<PeerTable>>,
    shutdown: Arc<AtomicBool>,
) {
    let _ = sock.set_read_timeout(Some(DISCOVERY_POLL));
    let announce = build_announce(&name, http_port);
    let broadcast_addr = format!("255.255.255.255:{udp_port}");
    // Announce immediately so peers learn about us without waiting a full
    // interval. send_to errors (e.g. no broadcast route) are non-fatal.
    let _ = sock.send_to(announce.as_bytes(), &broadcast_addr);
    let mut last_announce = Instant::now();
    let mut buf = [0u8; 512];
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match sock.recv_from(&mut buf) {
            Ok((n, src)) => {
                let msg = std::str::from_utf8(&buf[..n]).unwrap_or("");
                let mut table = peers.lock().unwrap();
                handle_datagram(msg, src, &name, &mut table);
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(e) => {
                tracing::warn!(error = %e, "mesh: UDP recv error; discovery loop exiting");
                break;
            }
        }
        if last_announce.elapsed() >= ANNOUNCE_INTERVAL {
            let _ = sock.send_to(announce.as_bytes(), &broadcast_addr);
            last_announce = Instant::now();
            peers.lock().unwrap().prune(Instant::now(), PEER_TTL);
        }
    }
}

/// Background HTTP server loop: accepts connections on `listener` (nonblocking
/// so it can poll `shutdown`) and hands each off to a short-lived handler
/// thread. Exits when `shutdown` is set.
fn http_loop(
    listener: TcpListener,
    name: String,
    peers: Arc<Mutex<PeerTable>>,
    shutdown: Arc<AtomicBool>,
) {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        match listener.accept() {
            Ok((stream, _)) => {
                let name = name.clone();
                let peers = Arc::clone(&peers);
                if std::thread::Builder::new()
                    .name("skadoosh-mesh-conn".to_string())
                    .spawn(move || handle_http(stream, &name, &peers))
                    .is_err()
                {
                    tracing::warn!("mesh: failed to spawn HTTP handler thread");
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                std::thread::sleep(HTTP_POLL);
            }
            Err(e) => {
                tracing::warn!(error = %e, "mesh: HTTP accept error; server loop exiting");
                break;
            }
        }
    }
}

/// Handles one HTTP/1.1 request. Recognizes `POST /forward` (echoes an
/// acknowledgment carrying this node's name and the forwarded summary, so the
/// forwarding node can relay it back to the user) and `GET /peers` (returns
/// the known peer names as a JSON array). Anything else gets a 404.
fn handle_http(mut stream: std::net::TcpStream, name: &str, peers: &Mutex<PeerTable>) {
    // Accepted sockets may inherit nonblocking from the listener on some
    // platforms; force blocking reads with a timeout for this connection.
    let _ = stream.set_nonblocking(false);
    let _ = stream.set_read_timeout(Some(Duration::from_secs(2)));

    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    let header_end = loop {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => return,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if let Some(pos) = find(&buf, b"\r\n\r\n") {
                    break pos + 4;
                }
                if buf.len() > HTTP_MAX_REQUEST {
                    return;
                }
            }
        }
    };

    let headers = String::from_utf8_lossy(&buf[..header_end]);
    let mut content_len = 0usize;
    let mut path = String::new();
    for (i, line) in headers.lines().enumerate() {
        if i == 0 {
            // Request line: "METHOD /path HTTP/1.1".
            path = line.split_whitespace().nth(1).unwrap_or("").to_string();
        } else if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_len = v.trim().parse().unwrap_or(0);
            }
        }
    }

    // Read the (remaining) body up to Content-Length.
    let body_start = header_end;
    while buf.len() - body_start < content_len {
        match stream.read(&mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.len() > HTTP_MAX_REQUEST {
                    return;
                }
            }
        }
    }
    let body_end = (body_start + content_len).min(buf.len());
    let body = &buf[body_start..body_end];

    let (status, content_type, response_body) = if path == "/forward" {
        let summary = serde_json::from_slice::<serde_json::Value>(body)
            .ok()
            .and_then(|v| {
                v.get("summary")
                    .and_then(|s| s.as_str())
                    .map(str::to_string)
            })
            .unwrap_or_default();
        (
            "200 OK",
            "text/plain",
            format!("[forwarded to {name}] {summary}"),
        )
    } else if path == "/peers" {
        let names = peers.lock().unwrap().names();
        let body = serde_json::to_string(&names).unwrap_or_else(|_| "[]".to_string());
        ("200 OK", "application/json", body)
    } else {
        ("404 Not Found", "text/plain", "not found".to_string())
    };

    let head = format!(
        "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n",
        response_body.len()
    );
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(response_body.as_bytes());
    let _ = stream.flush();
}

/// Finds the first occurrence of `needle` in `haystack`.
fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
