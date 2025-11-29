/// Compute cosine similarity between two embeddings.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<f32>();
    let norm_a_sq = a.iter().map(|x| x * x).sum::<f32>();
    let norm_b_sq = b.iter().map(|x| x * x).sum::<f32>();
    let denom = (norm_a_sq.sqrt() * norm_b_sq.sqrt()).max(1e-8);
    dot / denom
}
