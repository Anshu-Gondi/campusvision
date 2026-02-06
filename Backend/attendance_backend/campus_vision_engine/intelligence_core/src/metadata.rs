use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FaceMetadata {
    pub id: String,
    pub name: String,
    pub person_id: u64,
    pub roll_no: String,
    pub role: String,
    pub reliability: Option<f32>,
}
