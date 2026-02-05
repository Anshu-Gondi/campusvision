use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::Mutex;

/// Internal store of daily attendance per role
static DAILY_MARKED: Lazy<Mutex<HashMap<String, HashSet<usize>>>> = Lazy::new(|| {
    Mutex::new(HashMap::from([
        ("student".to_string(), HashSet::new()),
        ("teacher".to_string(), HashSet::new()),
    ]))
});

/// Marks a person as seen today. Returns `true` if they were newly marked.
pub fn mark_person_today(role: &str, id: usize) -> bool {
    let mut map = DAILY_MARKED.lock().unwrap();
    map.entry(role.to_string())
        .or_default()
        .insert(id)
}

/// Checks if a person is already marked today.
pub fn is_already_marked_today(role: &str, id: usize) -> bool {
    let map = DAILY_MARKED.lock().unwrap();
    map.get(role)
        .map(|set| set.contains(&id))
        .unwrap_or(false)
}

/// Clears all daily attendance records
pub fn clear_daily_records() {
    let mut map = DAILY_MARKED.lock().unwrap();
    for set in map.values_mut() {
        set.clear();
    }
}

/// Marks a TrackedFace ONLY if identity is stable and confident
/// Returns true if marking happened (newly marked)
pub fn mark_tracked_face(
    face: &super::tracker::TrackedFace,
    role: &str,
) -> bool {
    const MIN_CONFIDENCE: f32 = 0.85;

    if let Some(id) = face.person_id {
        if face.id_locked && face.confidence >= MIN_CONFIDENCE {
            return mark_person_today(role, id);
        }
    }
    false
}
