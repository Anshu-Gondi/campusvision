pub struct Decision {
    pub accept: bool,
    pub score: f32,
    pub reason: Option<&'static str>,
}

pub fn evaluate(q: &crate::quality::QualityMetrics, is_enrollment: bool) -> Decision {
    let blur_score = (q.blur / 150.0).min(1.0);
    let bright_score = ((q.brightness - 40.0) / 120.0).clamp(0.0, 1.0);
    let size_score = (q.face_area_ratio / 0.15).min(1.0);

    let score = 0.4 * blur_score + 0.3 * bright_score + 0.3 * size_score;

    let threshold = if is_enrollment { 0.65 } else { 0.5 };

    if score < threshold {
        let reason = if q.blur < 50.0 {
            "too_blurry"
        } else if q.brightness < 40.0 {
            "low_light"
        } else if q.face_area_ratio < 0.08 {
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

    Decision {
        accept: true,
        score,
        reason: None,
    }
}