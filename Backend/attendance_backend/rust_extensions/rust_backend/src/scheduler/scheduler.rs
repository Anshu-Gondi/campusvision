use crate::hnsw_helper::{count_by_role, get_metadata, search_in_role};
use chrono::NaiveTime;
use std::collections::HashMap;

#[derive(Clone)]
pub struct ClassRequest {
    pub class_name: String,
    pub section: String,
    pub subject: String,
    pub start_time: NaiveTime,
    pub end_time: NaiveTime,
}

#[derive(Clone)]
pub struct TeacherCandidate {
    pub id: usize,
    pub name: String,
    pub reliability: f32,
    pub workload: usize,
    pub similarity: f32,
    pub timetable: Vec<(NaiveTime, NaiveTime)>,
}

#[derive(Clone)]
pub struct Assignment {
    pub class: ClassRequest,
    pub teacher: TeacherCandidate,
}

/// Scheduler with adaptive heuristic Tree-AH search
pub struct FullScheduler {
    pub teacher_workload: HashMap<usize, usize>,
    pub teacher_timetable: HashMap<usize, Vec<(NaiveTime, NaiveTime)>>, // FIXED: timetable now global
}

impl FullScheduler {
    pub fn new() -> Self {
        Self {
            teacher_workload: HashMap::new(),
            teacher_timetable: HashMap::new(),
        }
    }

    pub fn assign_classes(&mut self, classes: &[ClassRequest]) -> Vec<Assignment> {
        // Tree-AH: sort by time (earlier classes branch first)
        let mut ordered = classes.to_vec();
        ordered.sort_by_key(|c| c.start_time);

        let mut current = Vec::new();
        let mut best = Vec::new();
        let mut best_score: f32 = -1.0;

        self.search(&ordered, 0, &mut current, &mut best, &mut best_score);
        best
    }

    fn search(
        &mut self,
        classes: &[ClassRequest],
        index: usize,
        current: &mut Vec<Assignment>,
        best: &mut Vec<Assignment>,
        best_score: &mut f32,
    ) {
        if index == classes.len() {
            let total_score = Self::compute_schedule_score(current);

            if total_score > *best_score {
                *best_score = total_score;
                *best = current.clone();
            }
            return;
        }

        let class = &classes[index];
        let candidates = self.get_candidates(class);

        // --- Pruning using optimistic bounds -------------------
        let current_score = Self::compute_schedule_score(current);
        let max_future = candidates.iter().map(|c| c.reliability * c.similarity).sum::<f32>();

        if current_score + max_future <= *best_score {
            return;
        }

        for mut teacher in candidates {
            if self.time_conflict(teacher.id, class) {
                continue;
            }

            // choose
            *self.teacher_workload.entry(teacher.id).or_insert(0) += 1;
            teacher.workload = self.teacher_workload[&teacher.id];

            // assign to timetable
            self.teacher_timetable
                .entry(teacher.id)
                .or_insert(vec![])
                .push((class.start_time, class.end_time));

            current.push(Assignment {
                class: class.clone(),
                teacher: teacher.clone(),
            });

            self.search(classes, index + 1, current, best, best_score);

            // backtrack
            current.pop();
            self.teacher_workload.get_mut(&teacher.id).map(|w| *w -= 1);

            let slots = self.teacher_timetable.get_mut(&teacher.id).unwrap();
            slots.pop();
        }
    }

    fn compute_schedule_score(assignments: &[Assignment]) -> f32 {
        assignments
            .iter()
            .map(|a| {
                // smoother diminishing returns: / (1 + ln(workload+1))
                let w = a.teacher.workload.max(1) as f32;
                a.teacher.reliability * a.teacher.similarity / (1.0 + w.ln())
            })
            .sum()
    }

    fn time_conflict(&self, teacher_id: usize, class: &ClassRequest) -> bool {
        if let Some(slots) = self.teacher_timetable.get(&teacher_id) {
            for &(s, e) in slots {
                if class.start_time < e && class.end_time > s {
                    return true;
                }
            }
        }
        false
    }

    /// Fetch top-K teachers using HNSW + workload penalty + reliability
    fn get_candidates(&self, class: &ClassRequest) -> Vec<TeacherCandidate> {
        let total = count_by_role("teacher");

        // get HNSW top 30 nearest teachers for this subject
        let subject_embedding = class.subject.as_bytes().iter().map(|b| *b as f32).collect::<Vec<f32>>();

        let hnsw_hits = search_in_role(&subject_embedding, "teacher", 30);

        let mut candidates = Vec::new();

        for (teacher_id, sim) in hnsw_hits {
            if let Some(meta) = get_metadata(teacher_id) {
                let workload = *self.teacher_workload.get(&teacher_id).unwrap_or(&0);

                let reliability = meta.reliability.unwrap_or(0.8);

                candidates.push(TeacherCandidate {
                    id: teacher_id,
                    name: meta.name.clone(),
                    reliability,
                    workload,
                    similarity: sim,
                    timetable: vec![],
                });
            }
        }

        // score = reliability * similarity / (1 + ln(workload+1))
        candidates.sort_by(|a, b| {
            let score_a = a.reliability * a.similarity / (1.0 + (a.workload.max(1) as f32).ln());
            let score_b = b.reliability * b.similarity / (1.0 + (b.workload.max(1) as f32).ln());
            score_b.partial_cmp(&score_a).unwrap()
        });

        candidates.into_iter().take(6).collect()
    }
}
