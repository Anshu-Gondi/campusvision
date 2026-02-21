use crate::scheduler::core::{
    ClassRequest,
    FullScheduler,
    GraphScheduler,
};
use chrono::NaiveTime;

/// Input DTO from Python (already parsed)
pub struct PyClassInput {
    pub class_name: String,
    pub section: String,
    pub subject: String,
    pub start_time: NaiveTime,
    pub end_time: NaiveTime,
}

/// Shared output structure
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

/// Normal scheduler
pub fn schedule_classes_rust(
    inputs: Vec<PyClassInput>,
    school_id: String,
) -> Vec<ScheduledClassResult> {
    let classes: Vec<ClassRequest> = inputs
        .into_iter()
        .map(|c| ClassRequest {
            class_name: c.class_name,
            section: c.section,
            subject: c.subject,
            start_time: c.start_time,
            end_time: c.end_time,
        })
        .collect();

    let embedding_dim = 32;
    let mut scheduler = FullScheduler::new(embedding_dim, school_id.clone());

    scheduler
        .assign_classes(&classes)
        .into_iter()
        .map(|r| ScheduledClassResult {
            class_name: r.class.class_name,
            section: r.class.section,
            subject: r.class.subject,
            teacher_id: r.teacher.id as u64,
            teacher_name: r.teacher.name,
            similarity: r.teacher.similarity,
            reliability: r.teacher.reliability,
            workload: r.teacher.workload as f32,
        })
        .collect()
}

/// Beam-search scheduler
pub fn schedule_classes_beam_rust(
    inputs: Vec<PyClassInput>,
    school_id: String,
) -> Vec<ScheduledClassResult> {
    let classes: Vec<ClassRequest> = inputs
        .into_iter()
        .map(|c| ClassRequest {
            class_name: c.class_name,
            section: c.section,
            subject: c.subject,
            start_time: c.start_time,
            end_time: c.end_time,
        })
        .collect();

    let embedding_dim = 32;

    let mut scheduler = GraphScheduler::new(
        embedding_dim,
        60,    // beam width
        0.02,  // similarity threshold
        school_id.clone(),
    );

    scheduler
        .assign_classes_beam(&classes)
        .into_iter()
        .map(|r| ScheduledClassResult {
            class_name: r.class.class_name,
            section: r.class.section,
            subject: r.class.subject,
            teacher_id: r.teacher.id as u64,
            teacher_name: r.teacher.name,
            similarity: r.teacher.similarity,
            reliability: r.teacher.reliability,
            workload: r.teacher.workload as f32,
        })
        .collect()
}
