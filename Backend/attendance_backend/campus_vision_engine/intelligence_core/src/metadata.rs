use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FaceMetadata {
    pub id: String,
    pub name: String,
    pub person_id: u64,
    pub roll_no: String,
    pub role: String,
    pub reliability: Option<f32>,
    pub deleted: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MetaHot {
    pub deleted: bool,
    pub role: u8, // 0 = student, 1 = teacher
    pub person_id: u64,
}
