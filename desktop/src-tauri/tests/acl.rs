//! Tauri 2 refuses any command the ACL does not allow ("Command … not allowed by ACL"), and only
//! at runtime. Every command handed to `generate_handler!` must be listed in build.rs (which
//! generates its permission) and granted in capabilities/default.json.

fn handlers() -> Vec<String> {
    let lib = include_str!("../src/lib.rs");
    let start = lib
        .find("generate_handler![")
        .expect("generate_handler! in lib.rs");
    let body = &lib[start + "generate_handler![".len()..];
    let body = &body[..body.find(']').expect("end of generate_handler!")];
    body.split(',')
        .map(|s| s.trim().rsplit("::").next().unwrap_or("").to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

#[test]
fn every_command_is_allowed() {
    let build = include_str!("../build.rs");
    let caps = include_str!("../capabilities/default.json");
    for cmd in handlers() {
        assert!(
            build.contains(&format!("\"{cmd}\"")),
            "{cmd} is missing from COMMANDS in build.rs"
        );
        let perm = format!("\"allow-{}\"", cmd.replace('_', "-"));
        assert!(
            caps.contains(&perm),
            "{perm} is missing from capabilities/default.json"
        );
    }
}
