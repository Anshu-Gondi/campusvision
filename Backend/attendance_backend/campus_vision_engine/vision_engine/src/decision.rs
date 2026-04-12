pub struct Decision {
    pub accept: bool,
    pub score: f32,
    pub reason: Option<&'static str>,
}

pub fn evaluate(q: &crate::quality::QualityMetrics, is_enrollment: bool) -> Decision {
    // =========================================================
    // 🔴 1. HARD REJECTION LAYER (NON-NEGOTIABLE)
    // =========================================================

    // Absolute garbage → reject immediately (save CPU + DB + embeddings)
    if q.blur < 40.0 {
        return Decision {
            accept: false,
            score: 0.0,
            reason: Some("too_blurry_hard"),
        };
    }

    if q.brightness < 25.0 {
        return Decision {
            accept: false,
            score: 0.0,
            reason: Some("extreme_low_light"),
        };
    }

    if q.face_area_ratio < 0.05 {
        return Decision {
            accept: false,
            score: 0.0,
            reason: Some("face_too_small_hard"),
        };
    }

    // Enrollment stricter hard rules
    if is_enrollment {
        if q.blur < 60.0 {
            return Decision {
                accept: false,
                score: 0.0,
                reason: Some("blur_not_allowed_enrollment"),
            };
        }

        if q.face_area_ratio < 0.08 {
            return Decision {
                accept: false,
                score: 0.0,
                reason: Some("face_too_small_enrollment"),
            };
        }
    }

    // =========================================================
    // 🟡 2. NORMALIZED SCORING LAYER
    // =========================================================

    // More stable normalization (avoids fake high scores)
    let blur_score = ((q.blur - 50.0) / 150.0).clamp(0.0, 1.0);
    let bright_score = ((q.brightness - 50.0) / 100.0).clamp(0.0, 1.0);
    let size_score = ((q.face_area_ratio - 0.05) / 0.15).clamp(0.0, 1.0);

    // Weighted score (tuned for embeddings reliability)
    let mut score = 0.0;

    score += 0.45 * blur_score.powf(1.2);
    score += 0.3 * bright_score.powf(1.1);
    score += 0.25 * size_score;

    // penalty system (this is what real systems do)
    if q.blur < 70.0 {
        score *= 0.75;
    }
    if q.brightness < 45.0 {
        score *= 0.8;
    }

    // =========================================================
    // 🟢 3. CONTEXT-AWARE THRESHOLD
    // =========================================================

    let threshold = if is_enrollment { 0.7 } else { 0.52 };

    if score < threshold {
        let reason = if q.blur < 70.0 {
            "too_blurry"
        } else if q.brightness < 50.0 {
            "low_light"
        } else if q.face_area_ratio < 0.1 {
            "face_too_small"
        } else {
            "low_quality"
        };

        return Decision {
            accept: false,
            score,
            reason: Some(reason),
        };
    }

    // =========================================================
    // ✅ ACCEPT
    // =========================================================

    Decision {
        accept: true,
        score,
        reason: None,
    }
}
