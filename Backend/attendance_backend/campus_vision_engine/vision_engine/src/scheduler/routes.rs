use axum::{extract::State, routing::post, Json, Router};
use chrono::NaiveTime;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use intelligence_core::scheduler::core::{ClassRequest, FullScheduler, GraphScheduler};

use crate::app::AppState;

//
// ────────────────────────────────
// 1️⃣ Request DTOs (HTTP layer)
// ────────────────────────────────
//

#[derive(Deserialize)]
pub struct ScheduleClassInput {
    pub class_name: String,
    pub section: String,
    pub subject: String,
    pub start_time: NaiveTime,
    pub end_time: NaiveTime,
}

#[derive(Deserialize)]
pub struct ScheduleRequest {
    pub classes: Vec<ScheduleClassInput>,
    /// "normal" (default) | "beam"
    pub mode: Option<String>,
}

//
// ────────────────────────────────
// 2️⃣ Response DTOs (HTTP layer)
// ────────────────────────────────
//

#[derive(Serialize)]
pub struct ScheduledClassResult {
    pub class_name: String,
    pub section: String,
    pub subject: String,
    pub teacher_id: u64,
    pub teacher_name: String,
    pub similarity: f32,
    pub reliability: f32,
    pub workload: f32,
}

//
// ────────────────────────────────
// 3️⃣ Axum handler
// ────────────────────────────────
//

pub async fn schedule_classes_handler(
    State(_state): State<Arc<AppState>>,
    Json(payload): Json<ScheduleRequest>,
) -> Json<Vec<ScheduledClassResult>> {
    // ── Convert HTTP DTO → core input
    let classes: Vec<ClassRequest> = payload
        .classes
        .iter()
        .map(|c| ClassRequest {
            class_name: c.class_name.clone(),
            section: c.section.clone(),
            subject: c.subject.clone(),
            start_time: c.start_time,
            end_time: c.end_time,
        })
        .collect();

    let embedding_dim = 32;

    // ── Select scheduler strategy
    let assignments = match payload.mode.as_deref() {
        Some("beam") => {
            let mut scheduler = GraphScheduler::new(embedding_dim, 60, 0.02);
            scheduler.assign_classes_beam(&classes)
        }
        _ => {
            let mut scheduler = FullScheduler::new(embedding_dim);
            scheduler.assign_classes(&classes)
        }
    };

    // ── Convert core output → HTTP response
    let response = assignments
        .iter() // borrow, does NOT move
        .map(|r| ScheduledClassResult {
            class_name: r.class.class_name.clone(),
            section: r.class.section.clone(),
            subject: r.class.subject.clone(),
            teacher_id: r.teacher.id as u64,
            teacher_name: r.teacher.name.clone(),
            similarity: r.teacher.similarity,
            reliability: r.teacher.reliability,
            workload: r.teacher.workload as f32,
        })
        .collect();

    Json(response)
}

//
// ────────────────────────────────
// 4️⃣ Route registration
// ────────────────────────────────
//

pub fn scheduler_routes(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/schedule/classes", post(schedule_classes_handler))
        .with_state(state)
}
