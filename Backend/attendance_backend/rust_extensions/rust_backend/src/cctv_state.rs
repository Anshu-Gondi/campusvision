// src/cctv_state.rs
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

static DAILY_MARKED: Lazy<Mutex<HashMap<String, HashSet<usize>>>> = Lazy::new(|| {
    Mutex::new(HashMap::from([
        ("student".to_string(), HashSet::new()),
        ("teacher".to_string(), HashSet::new()),
    ]))
});

pub fn mark_person_today(role: &str, id: usize) -> bool {
    let mut map = DAILY_MARKED.lock().unwrap();
    map.entry(role.to_string())
        .or_default()
        .insert(id)
}

pub fn is_already_marked_today(role: &str, id: usize) -> bool {
    let map = DAILY_MARKED.lock().unwrap();
    map.get(role)
        .map(|set| set.contains(&id))
        .unwrap_or(false)
}

pub fn clear_daily_records() {
    let mut map = DAILY_MARKED.lock().unwrap();
    for set in map.values_mut() {
        set.clear();
    }
}