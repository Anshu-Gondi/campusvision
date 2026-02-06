use intelligence_core::embeddings::{get_metadata, search_in_role};
use chrono::NaiveTime;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

const MAX_SUBJECT_CACHE: usize = 256;

#[derive(Clone)]
pub struct ScheduleLimits {
    pub max_duration: Duration, // hard timeout
    pub max_expansions: usize,  // DFS expansion limit
    pub max_depth: usize,       // recursion bound
}

impl Default for ScheduleLimits {
    fn default() -> Self {
        Self {
            max_duration: Duration::from_millis(200), // Django-safe
            max_expansions: 25_000,
            max_depth: 64,
        }
    }
}

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

/* ----------------------------
   Embedding helper (deterministic)
   ----------------------------
   Creates a fixed-dim embedding from text using multiple hash seeds.
   The vector is L2-normalized so it works nicely with cosine-based HNSW.
   Replace this with a real embedding model later.
*/
pub fn hash_to_embedding(text: &str, dim: usize) -> Vec<f32> {
    // Use several different hash seeds by hashing (text + idx) to spread info.
    let mut vec = vec![0f32; dim];
    for (i, ch) in text.chars().enumerate() {
        // lightweight feature: char code & position
        let bucket = (i * 179 + (ch as usize)) % dim;
        vec[bucket] += (ch as u32) as f32 * 0.001_f32; // scale down
    }

    // Add a few hashed perturbations for more spread
    for seed in 0..3 {
        let mut hasher = DefaultHasher::new();
        // combine seed for variety
        (text, seed).hash(&mut hasher);
        let h = hasher.finish();
        for d in 0..dim {
            // mix bits into float
            let v = (((h >> (d % 64)) & 0xFF) as i32 - 128) as f32;
            vec[d % dim] += v * 0.0005_f32;
        }
    }

    // Normalize (L2)
    let sum_sq: f32 = vec.iter().map(|v| v * v).sum();
    let norm = sum_sq.sqrt().max(1e-6_f32);
    vec.iter_mut().for_each(|v| *v /= norm);
    vec
}

/* ----------------------------
   Candidate cache to avoid repeated HNSW calls per subject
   Now stores only ranked (teacher_id, similarity) lists.
   Filtering by timetable and workload is done at selection time.
*/
#[derive(Default)]
pub struct CandidateCache {
    // subject -> ranked list of (teacher_id, similarity)
    pub map: HashMap<String, Vec<(usize, f32)>>,
    pub dim: usize,
}

impl CandidateCache {
    pub fn new(dim: usize) -> Self {
        Self {
            map: HashMap::new(),
            dim,
        }
    }

    /// Ensure we have the ranking for `subject` cached; returns ranked (id, sim).
    pub fn ranked_for_subject(&mut self, subject: &str, k: usize) -> Vec<(usize, f32)> {
        if let Some(r) = self.map.get(subject) {
            return r.clone();
        }

        let embedding = hash_to_embedding(subject, self.dim);
        let hits = search_in_role(&embedding, "teacher", k);
        // store and return
        if self.map.len() >= MAX_SUBJECT_CACHE {
            // simple eviction: remove a random key (not LRU, but simple)
            if let Some(first_key) = self.map.keys().next().cloned() {
                self.map.remove(&first_key);
            }
        }
        self.map.insert(subject.to_string(), hits.clone());
        hits
    }

    /// Build TeacherCandidate list filtered by provided timetable and workload.
    /// - `timetable_map` maps teacher_id -> Vec<(start,end)> representing already-assigned slots.
    /// - We skip teachers who conflict with `class_req`.
    /// - We return up to `take_k` candidates sorted by score (reliability*similarity/(1+ln(workload+1))).
    pub fn filtered_candidates(
        &mut self,
        subject: &str,
        k: usize,
        workload_map: &HashMap<usize, usize>,
        timetable_map: &HashMap<usize, Vec<(NaiveTime, NaiveTime)>>,
        class_req: &ClassRequest,
        take_k: usize,
    ) -> Vec<TeacherCandidate> {
        let ranked = self.ranked_for_subject(subject, k);

        let mut candidates: Vec<TeacherCandidate> = Vec::new();

        for (teacher_id, sim) in ranked.into_iter() {
            // get metadata
            if let Some(meta) = get_metadata(teacher_id) {
                // check timetable conflict against provided timetable_map
                if let Some(slots) = timetable_map.get(&teacher_id) {
                    let mut conflict = false;
                    for &(s, e) in slots {
                        if class_req.start_time < e && class_req.end_time > s {
                            conflict = true;
                            break;
                        }
                    }
                    if conflict {
                        continue; // teacher busy for this class time
                    }
                }

                let workload = *workload_map.get(&teacher_id).unwrap_or(&0);
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

        // sort by full score and take top
        candidates.sort_by(|a, b| {
            let score_a = a.reliability * a.similarity / (1.0 + (a.workload.max(1) as f32).ln());
            let score_b = b.reliability * b.similarity / (1.0 + (b.workload.max(1) as f32).ln());
            score_b.partial_cmp(&score_a).unwrap()
        });

        candidates.into_iter().take(take_k).collect()
    }
}

/* ----------------------------
   FullScheduler (improved Tree-AH backtracking)
   Uses CandidateCache::filtered_candidates to ensure availability.
*/
pub struct FullScheduler {
    pub teacher_workload: HashMap<usize, usize>,
    pub teacher_timetable: HashMap<usize, Vec<(NaiveTime, NaiveTime)>>,
    // small cache embedded in scheduler instance
    pub cache: CandidateCache,
    start_time: Instant,
    expansions: usize,
}

impl FullScheduler {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            teacher_workload: HashMap::new(),
            teacher_timetable: HashMap::new(),
            cache: CandidateCache::new(embedding_dim),
            start_time: Instant::now(),
            expansions: 0,
        }
    }

    pub fn assign_classes(&mut self, classes: &[ClassRequest]) -> Vec<Assignment> {
        let limits = ScheduleLimits::default();
        self.assign_classes_limited(classes, &limits)
    }

    pub fn assign_classes_limited(
        &mut self,
        classes: &[ClassRequest],
        limits: &ScheduleLimits,
    ) -> Vec<Assignment> {
        self.teacher_workload.clear();
        self.teacher_timetable.clear();
        self.start_time = Instant::now();
        self.expansions = 0;

        let mut ordered = classes.to_vec();
        ordered.sort_by_key(|c| c.start_time);

        let mut current = Vec::new();
        let mut best = Vec::new();
        let mut best_score = -1.0;

        self.search_limited(
            &ordered,
            0,
            &mut current,
            &mut best,
            &mut best_score,
            limits,
        );

        best
    }

    fn search_limited(
        &mut self,
        classes: &[ClassRequest],
        index: usize,
        current: &mut Vec<Assignment>,
        best: &mut Vec<Assignment>,
        best_score: &mut f32,
        limits: &ScheduleLimits,
    ) {
        // ⏱ timeout
        if self.start_time.elapsed() > limits.max_duration {
            return;
        }

        // 🔢 expansion guard
        self.expansions += 1;
        if self.expansions > limits.max_expansions {
            return;
        }

        if index == classes.len() || index >= limits.max_depth {
            let score = Self::compute_schedule_score(current);
            if score > *best_score {
                *best_score = score;
                *best = current.clone();
            }
            return;
        }

        let class = &classes[index];

        let candidates = self.cache.filtered_candidates(
            &class.subject,
            30,
            &self.teacher_workload,
            &self.teacher_timetable,
            class,
            6,
        );

        let current_score = Self::compute_schedule_score(current);
        let optimistic = candidates
            .first()
            .map(|c| c.reliability * c.similarity)
            .unwrap_or(0.0)
            * (classes.len() - index) as f32;

        if current_score + optimistic <= *best_score {
            return;
        }

        for mut teacher in candidates {
            if self.time_conflict(teacher.id, class) {
                continue;
            }

            *self.teacher_workload.entry(teacher.id).or_insert(0) += 1;
            teacher.workload = self.teacher_workload[&teacher.id];

            self.teacher_timetable
                .entry(teacher.id)
                .or_default()
                .push((class.start_time, class.end_time));

            current.push(Assignment {
                class: class.clone(),
                teacher: teacher.clone(),
            });

            self.search_limited(classes, index + 1, current, best, best_score, limits);

            current.pop();
            *self.teacher_workload.get_mut(&teacher.id).unwrap() -= 1;
            self.teacher_timetable.get_mut(&teacher.id).unwrap().pop();
        }
    }

    fn compute_schedule_score(assignments: &[Assignment]) -> f32 {
        assignments
            .iter()
            .map(|a| {
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
}

/* ----------------------------
Graph-AH / Beam-search scheduler (fast & practical)
Uses CandidateCache::filtered_candidates so each beam state only
considers teachers that are free in that state's timetable.
----------------------------*/
pub struct GraphScheduler {
    pub teacher_workload: HashMap<usize, usize>,
    pub teacher_timetable: HashMap<usize, Vec<(NaiveTime, NaiveTime)>>,
    pub cache: CandidateCache,
    pub beam_width: usize,
    pub fairness_penalty: f32,
}

#[derive(Clone)]
struct BeamState {
    pub assignments: Vec<AssignmentIdx>,
    pub workload: HashMap<usize, usize>,
    pub timetable: HashMap<usize, Vec<(NaiveTime, NaiveTime)>>,
    pub score: f32,
}

#[derive(Clone, Copy)]
struct AssignmentIdx {
    class_idx: usize,
    teacher_id: usize,
}

impl GraphScheduler {
    pub fn new(embedding_dim: usize, beam_width: usize, fairness_penalty: f32) -> Self {
        Self {
            teacher_workload: HashMap::new(),
            teacher_timetable: HashMap::new(),
            cache: CandidateCache::new(embedding_dim),
            beam_width,
            fairness_penalty,
        }
    }

    pub fn assign_classes_beam(&mut self, classes: &[ClassRequest]) -> Vec<Assignment> {
        let mut ordered: Vec<usize> = (0..classes.len()).collect();
        ordered.sort_by_key(|&i| classes[i].start_time);

        let mut beam = vec![BeamState {
            assignments: Vec::with_capacity(classes.len()),
            workload: HashMap::new(),
            timetable: HashMap::new(),
            score: 0.0,
        }];

        for &class_idx in &ordered {
            let class = &classes[class_idx];
            let mut next_beam = Vec::with_capacity(self.beam_width * 4);

            for state in beam.iter() {
                let candidates = self.cache.filtered_candidates(
                    &class.subject,
                    30,
                    &state.workload,
                    &state.timetable,
                    class,
                    6,
                );

                for cand in candidates {
                    if Self::time_conflict(&state.timetable, cand.id, class) {
                        continue;
                    }

                    let mut new_state = BeamState {
                        assignments: state.assignments.clone(), // small Vec<AssignmentIdx>
                        workload: state.workload.clone(),
                        timetable: state.timetable.clone(),
                        score: state.score,
                    };

                    let w = *new_state.workload.get(&cand.id).unwrap_or(&0) + 1;
                    new_state.workload.insert(cand.id, w);
                    new_state
                        .timetable
                        .entry(cand.id)
                        .or_default()
                        .push((class.start_time, class.end_time));

                    let inc = cand.reliability * cand.similarity / (1.0 + (w as f32).ln());
                    new_state.score += inc;

                    new_state.assignments.push(AssignmentIdx {
                        class_idx,
                        teacher_id: cand.id,
                    });

                    next_beam.push(new_state);
                }
            }

            next_beam.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
            next_beam.truncate(self.beam_width);

            beam = next_beam;
            if beam.is_empty() {
                break;
            }
        }

        self.resolve_assignments(&beam, classes)
    }

    fn resolve_assignments(&self, beam: &[BeamState], classes: &[ClassRequest]) -> Vec<Assignment> {
        let best = beam.iter().max_by(|a, b| {
            a.score
                .partial_cmp(&b.score)
                .unwrap()
                .then(a.assignments.len().cmp(&b.assignments.len()))
        });

        let Some(state) = best else { return vec![] };

        state
            .assignments
            .iter()
            .filter_map(|a| {
                let meta = get_metadata(a.teacher_id)?;
                Some(Assignment {
                    class: classes[a.class_idx].clone(), // one-time clone
                    teacher: TeacherCandidate {
                        id: a.teacher_id,
                        name: meta.name.clone(),
                        reliability: meta.reliability.unwrap_or(0.8),
                        workload: 0,
                        similarity: 0.0,
                        timetable: vec![],
                    },
                })
            })
            .collect()
    }

    fn time_conflict(
        timetable: &HashMap<usize, Vec<(NaiveTime, NaiveTime)>>,
        teacher_id: usize,
        class: &ClassRequest,
    ) -> bool {
        if let Some(slots) = timetable.get(&teacher_id) {
            for &(s, e) in slots {
                if class.start_time < e && class.end_time > s {
                    return true;
                }
            }
        }
        false
    }
}

/* ----------------------------
   Notes / How to use
   ----------------------------
   - CandidateCache now caches only ranked (teacher_id, similarity).
     Availability filtering happens when building candidates for a
     specific timetable & class time, so busy teachers are excluded.
   - For FullScheduler, we filtered against the scheduler's global timetable.
   - For GraphScheduler, each beam state's timetable is used for filtering,
     so each beam branch only expands with truly available teachers.
   - Recommended: embedding dim 32 or 64, beam_width 50..200 depending on scale.
*/
