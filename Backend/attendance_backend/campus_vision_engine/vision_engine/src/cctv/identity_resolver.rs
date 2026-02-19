use intelligence_core::embeddings::batch_search;
use std::sync::Arc;

#[derive(Clone)]
pub struct IdentityResolver;

impl IdentityResolver {

    pub fn resolve(
        embedding: &Arc<Vec<f32>>,
    ) -> Option<(usize, f32)> {

        let student = batch_search(
            std::slice::from_ref(embedding),
            "student",
            1
        );

        let teacher = batch_search(
            std::slice::from_ref(embedding),
            "teacher",
            1
        );

        match (student.get(0), teacher.get(0)) {
            (Some(Some(s)), Some(Some(t))) => {
                if s.1 >= t.1 { Some(*s) } else { Some(*t) }
            }
            (Some(Some(s)), _) => Some(*s),
            (_, Some(Some(t))) => Some(*t),
            _ => None,
        }
    }
}
