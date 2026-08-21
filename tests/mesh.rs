//! Multi-agent mesh integration tests: announce parsing/round-trip, peer
//! table management, mock UDP discovery (feeding datagrams through
//! `handle_datagram` and over real loopback UDP), and the embedded HTTP
//! forward server.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, UdpSocket};
use std::time::{Duration, Instant};

use skadoosh::config::Config;
use skadoosh::forward::{mesh_forward_tool_definition, FORWARD_TOOL_NAME};
use skadoosh::mesh::{
    build_announce, handle_datagram, parse_announce, MeshNode, PeerTable, DEFAULT_MESH_PORT,
};

// ---------------------------------------------------------------------------
// Announce parsing / building
// ---------------------------------------------------------------------------

/// A well-formed announce parses to `(name, http_port)`.
#[test]
fn parse_announce_valid() {
    assert_eq!(
        parse_announce("skadoosh-mesh alice 9876"),
        Some(("alice".to_string(), 9876))
    );
    // Leading/trailing whitespace is tolerated.
    assert_eq!(
        parse_announce("  skadoosh-mesh bob 1234  "),
        Some(("bob".to_string(), 1234))
    );
}

/// Malformed or foreign datagrams are rejected.
#[test]
fn parse_announce_rejects_bad_input() {
    assert_eq!(parse_announce("hello"), None);
    assert_eq!(parse_announce("skadoosh-mesh alice"), None); // missing port
    assert_eq!(parse_announce("skadoosh-mesh alice notaport"), None);
    assert_eq!(parse_announce(""), None);
}

/// `build_announce` round-trips through `parse_announce`.
#[test]
fn build_and_parse_round_trip() {
    let msg = build_announce("node-42", 9876);
    assert_eq!(parse_announce(&msg), Some(("node-42".to_string(), 9876)));
    assert!(msg.starts_with("skadoosh-mesh "));
}

// ---------------------------------------------------------------------------
// Peer table management
// ---------------------------------------------------------------------------

/// Recording a peer makes it lookup-able by name and endpoint.
#[test]
fn peer_table_record_and_lookup() {
    let now = Instant::now();
    let mut table = PeerTable::new();
    assert!(table.is_empty());
    table.record("alice", "192.168.1.5:9876", now);
    assert_eq!(table.len(), 1);
    assert_eq!(table.lookup("alice").as_deref(), Some("192.168.1.5:9876"));
    assert!(table.lookup("bob").is_none(), "unknown peer is None");
}

/// Re-recording a peer refreshes its endpoint in place.
#[test]
fn peer_table_record_refreshes_endpoint() {
    let now = Instant::now();
    let mut table = PeerTable::new();
    table.record("alice", "10.0.0.1:9876", now);
    table.record("alice", "10.0.0.9:9999", now);
    assert_eq!(table.len(), 1, "same name overwrites, not appended");
    assert_eq!(table.lookup("alice").as_deref(), Some("10.0.0.9:9999"));
}

/// `prune` drops peers not seen within the TTL, keeping fresh ones.
#[test]
fn peer_table_prune_drops_stale_peers() {
    let base = Instant::now();
    let mut table = PeerTable::new();
    table.record("fresh", "1.1.1.1:1", base);
    // "stale" was last seen 60s ago.
    table.record("stale", "2.2.2.2:2", base - Duration::from_secs(60));
    // `Instant - Duration` is fine for a past timestamp used as last_seen.
    table.prune(base, Duration::from_secs(30));
    assert_eq!(table.lookup("fresh").as_deref(), Some("1.1.1.1:1"));
    assert!(
        table.lookup("stale").is_none(),
        "stale peer should have been pruned"
    );
}

/// `names` returns the peer names sorted.
#[test]
fn peer_table_names_sorted() {
    let now = Instant::now();
    let mut table = PeerTable::new();
    table.record("charlie", "1.1.1.1:1", now);
    table.record("alice", "2.2.2.2:2", now);
    table.record("bob", "3.3.3.3:3", now);
    assert_eq!(table.names(), vec!["alice", "bob", "charlie"]);
}

// ---------------------------------------------------------------------------
// Mock UDP discovery (pure path)
// ---------------------------------------------------------------------------

/// Feeding a mock announce datagram through `handle_datagram` records the
/// peer at `{src.ip}:{announced_http_port}`.
#[test]
fn handle_datagram_records_peer_from_mock_datagram() {
    let src: SocketAddr = "192.168.1.7:30000".parse().unwrap();
    let mut table = PeerTable::new();
    handle_datagram("skadoosh-mesh alice 9876", src, "self", &mut table);
    assert_eq!(table.lookup("alice").as_deref(), Some("192.168.1.7:9876"));
}

/// A node ignores its own announcements.
#[test]
fn handle_datagram_ignores_self_announce() {
    let src: SocketAddr = "192.168.1.7:30000".parse().unwrap();
    let mut table = PeerTable::new();
    handle_datagram("skadoosh-mesh self 9876", src, "self", &mut table);
    assert!(table.is_empty(), "self-announce must not be recorded");
}

/// Malformed datagrams are silently ignored.
#[test]
fn handle_datagram_ignores_malformed() {
    let src: SocketAddr = "192.168.1.7:30000".parse().unwrap();
    let mut table = PeerTable::new();
    handle_datagram("not-a-mesh-message", src, "self", &mut table);
    assert!(table.is_empty());
}

// ---------------------------------------------------------------------------
// Embedded HTTP forward server
// ---------------------------------------------------------------------------

/// Starts a mesh node on an OS-assigned port, panicking if it could not bind
/// the HTTP server.
fn start_node(name: &str) -> MeshNode {
    let node = MeshNode::start(name, 0);
    assert!(node.http_addr().is_some(), "mesh HTTP server should bind");
    node
}

/// Sends a blocking HTTP request and returns the response body (text after
/// the blank header/body separator).
fn http_request(addr: SocketAddr, method: &str, path: &str, body: &str) -> String {
    let mut stream =
        TcpStream::connect_timeout(&addr, Duration::from_secs(2)).expect("connect to mesh HTTP");
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .expect("set read timeout");
    let head = if body.is_empty() {
        format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nconnection: close\r\n\r\n")
    } else {
        format!(
            "{method} {path} HTTP/1.1\r\nHost: {addr}\r\
             content-type: application/json\r\ncontent-length: {}\r\n\
             connection: close\r\n\r\n{body}",
            body.len()
        )
    };
    stream.write_all(head.as_bytes()).expect("write request");
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).expect("read response");
    let s = String::from_utf8_lossy(&buf).into_owned();
    match s.find("\r\n\r\n") {
        Some(p) => s[p + 4..].to_string(),
        None => s,
    }
}

/// `POST /forward` replies with an acknowledgment carrying the node's name
/// and the forwarded summary.
#[test]
fn http_forward_endpoint_replies_with_acknowledgment() {
    let node = start_node("alice");
    let addr = node.http_addr().unwrap();
    let body = r#"{"reason":"out of scope","summary":"check invoice status","current_query":"status?","history":[]}"#;
    let resp = http_request(addr, "POST", "/forward", body);
    assert!(
        resp.contains("[forwarded to alice]"),
        "response should name the receiving node: {resp}"
    );
    assert!(
        resp.contains("check invoice status"),
        "response should echo the summary: {resp}"
    );
}

/// `GET /peers` returns a JSON array of known peer names.
#[test]
fn http_peers_endpoint_returns_json_names() {
    let node = start_node("alice");
    // Seed the peer table directly (no real peer needed).
    {
        let mut table = node.peer_table().lock().unwrap();
        table.record("bob", "10.0.0.2:9876", Instant::now());
        table.record("carol", "10.0.0.3:9876", Instant::now());
    }
    let addr = node.http_addr().unwrap();
    let resp = http_request(addr, "GET", "/peers", "");
    let names: Vec<String> =
        serde_json::from_str(&resp).expect("/peers should return a JSON array");
    assert_eq!(names, vec!["bob".to_string(), "carol".to_string()]);
}

/// An unknown path returns a 404 body.
#[test]
fn http_unknown_path_returns_not_found() {
    let node = start_node("alice");
    let addr = node.http_addr().unwrap();
    let resp = http_request(addr, "GET", "/nope", "");
    assert_eq!(resp, "not found");
}

// ---------------------------------------------------------------------------
// Loopback UDP discovery (real socket, single machine)
// ---------------------------------------------------------------------------

/// A real mesh node discovers a peer whose announcement is sent over loopback
/// UDP to the node's bound discovery socket.
#[test]
fn loopback_udp_discovery_records_peer() {
    let node = start_node("self-node");
    let udp_addr = node
        .udp_addr()
        .expect("mesh UDP socket should bind for discovery test");

    // A separate socket on loopback plays the announcing peer.
    let sender = UdpSocket::bind("127.0.0.1:0").expect("bind sender socket");
    let src_port: u16 = 4242; // the (fake) HTTP port the peer claims
    let announce = build_announce("remote-peer", src_port);
    sender
        .send_to(announce.as_bytes(), udp_addr)
        .expect("send announce datagram");

    // Poll the peer table: the discovery loop reads with a 500 ms timeout, so
    // the peer appears within a bounded window.
    let deadline = Instant::now() + Duration::from_secs(3);
    let mut found = None;
    while Instant::now() < deadline {
        if let Some(endpoint) = node.peer_table().lock().unwrap().lookup("remote-peer") {
            found = Some(endpoint);
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    let endpoint = found.expect("peer should have been discovered via UDP");
    assert!(
        endpoint.ends_with(&format!(":{src_port}")),
        "endpoint should carry the announced HTTP port: {endpoint}"
    );
    assert!(
        endpoint.starts_with("127.0.0.1:"),
        "endpoint should use the loopback source IP: {endpoint}"
    );
}

// ---------------------------------------------------------------------------
// Config surface
// ---------------------------------------------------------------------------

/// The new mesh flags parse from the CLI with their defaults.
#[test]
fn mesh_flags_parse_from_cli() {
    use clap::Parser;
    let config = Config::try_parse_from([
        "skadoosh",
        "--mesh",
        "--mesh-port",
        "7000",
        "--agent-name",
        "desk-agent",
    ])
    .expect("parses");
    assert!(config.mesh);
    assert_eq!(config.mesh_port, 7000);
    assert_eq!(config.agent_name.as_deref(), Some("desk-agent"));

    // Defaults (no flags): mesh off, default port, no agent name.
    let default = Config::try_parse_from(["skadoosh"]).expect("parses");
    assert!(!default.mesh);
    assert_eq!(default.mesh_port, DEFAULT_MESH_PORT);
    assert!(default.agent_name.is_none());
}

/// `Config::default()` matches the clap defaults for the mesh fields.
#[test]
fn default_config_mesh_fields_match_clap() {
    use clap::Parser;
    let from_clap = Config::try_parse_from(["skadoosh"]).expect("parses");
    let default = Config::default();
    assert_eq!(from_clap.mesh, default.mesh);
    assert_eq!(from_clap.mesh_port, default.mesh_port);
    assert_eq!(from_clap.agent_name, default.agent_name);
    assert!(!default.mesh, "mesh is off by default");
    assert_eq!(default.mesh_port, DEFAULT_MESH_PORT);
}

// ---------------------------------------------------------------------------
// Mesh-extended forward tool definition
// ---------------------------------------------------------------------------

/// The mesh `forward_call` tool definition adds an optional `target` argument.
#[test]
fn mesh_forward_tool_definition_adds_target() {
    let tool = mesh_forward_tool_definition();
    assert_eq!(tool.function.name, FORWARD_TOOL_NAME);
    let params = &tool.function.parameters;
    assert_eq!(params["type"], "object");
    assert_eq!(params["properties"]["target"]["type"], "string");
    assert_eq!(params["properties"]["reason"]["type"], "string");
    assert_eq!(params["properties"]["summary"]["type"], "string");
    // target is optional; only reason + summary are required.
    let required = params["required"].as_array().expect("required is array");
    let names: Vec<&str> = required.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"reason"));
    assert!(names.contains(&"summary"));
    assert!(
        !names.contains(&"target"),
        "target must be optional, not required"
    );
}
