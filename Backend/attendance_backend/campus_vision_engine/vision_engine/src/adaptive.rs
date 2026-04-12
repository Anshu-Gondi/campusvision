#[derive(Debug, Clone, Copy)]
pub enum QualityTier {
    High,
    Medium,
    Low,
    Reject,
}

use crate::decision::Decision;

pub fn map_decision(decision: &Decision) -> QualityTier {
    let s = decision.score;

    if s >= 0.80 {
        QualityTier::High
    } else if s >= 0.65 {
        QualityTier::Medium
    } else if s >= 0.50 {
        QualityTier::Low
    } else {
        QualityTier::Reject
    }
}

pub struct AdaptiveResult {
    pub proceed: bool,
    pub warning: Option<String>,
    pub tier: QualityTier,
}

pub fn adapt(decision: &Decision, is_enrollment: bool) -> AdaptiveResult {
    let tier = map_decision(&decision);

    match tier {
        // =========================================================
        // 🟢 HIGH QUALITY → ALWAYS PROCEED
        // =========================================================
        QualityTier::High => AdaptiveResult {
            proceed: true,
            warning: None,
            tier,
        },

        // =========================================================
        // 🟡 MEDIUM QUALITY → SAFE TO PROCEED
        // =========================================================
        QualityTier::Medium => AdaptiveResult {
            proceed: true,
            warning: None,
            tier,
        },

        // =========================================================
        // 🟠 LOW QUALITY → CONTEXT DEPENDENT
        // =========================================================
        QualityTier::Low => {
            if is_enrollment {
                // ❌ Enrollment must be strict
                return AdaptiveResult {
                    proceed: false,
                    warning: Some(format!(
                        "Enrollment rejected: {}",
                        decision.reason.unwrap_or("low_quality")
                    )),
                    tier,
                };
            }

            // ✅ Attendance can tolerate
            AdaptiveResult {
                proceed: true,
                warning: Some(format!(
                    "Low quality image: {}",
                    decision.reason.unwrap_or("unknown")
                )),
                tier,
            }
        }

        // =========================================================
        // 🔴 REJECT → NEVER PROCEED
        // =========================================================
        QualityTier::Reject => AdaptiveResult {
            proceed: false,
            warning: Some(format!(
                "Image rejected: {}",
                decision.reason.unwrap_or("unknown")
            )),
            tier,
        },
    }
}