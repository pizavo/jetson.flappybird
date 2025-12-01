use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Represents the state of the game
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
pub struct GameState {
    pub bird_y: f32,
    pub bird_velocity: f32,
    pub bird_x: f32,
    pub pipes: Vec<Pipe>,
    pub score: i32,
    pub alive: bool,
    pub frame_count: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "python", pyclass)]
pub struct Pipe {
    pub x: f32,
    pub gap_y: f32,
    pub gap_height: f32,
    pub width: f32,
    pub passed: bool,
}

#[cfg(feature = "python")]
#[pymethods]
impl GameState {
    #[new]
    fn new() -> Self {
        GameState::default()
    }

    fn to_json(&self) -> PyResult<String> {
        serde_json::to_string(self).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Serialization error: {}", e))
        })
    }

    #[staticmethod]
    fn from_json(json: &str) -> PyResult<Self> {
        serde_json::from_str(json).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Deserialization error: {}", e))
        })
    }

    fn get_observation(&self) -> Vec<f32> {
        let mut obs = vec![self.bird_y, self.bird_velocity];

        // Find the next pipe
        if let Some(next_pipe) = self.pipes.iter().find(|p| p.x + p.width > self.bird_x) {
            obs.push(next_pipe.x - self.bird_x);
            obs.push(next_pipe.gap_y);
            obs.push(next_pipe.gap_y + next_pipe.gap_height);
        } else {
            obs.push(800.0);
            obs.push(300.0);
            obs.push(500.0);
        }

        obs
    }

    fn get_score(&self) -> i32 {
        self.score
    }

    fn is_alive(&self) -> bool {
        self.alive
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl Pipe {
    #[new]
    fn new(x: f32, gap_y: f32, gap_height: f32, width: f32) -> Self {
        Pipe {
            x,
            gap_y,
            gap_height,
            width,
            passed: false,
        }
    }
}

impl Default for GameState {
    fn default() -> Self {
        GameState {
            bird_y: 300.0,
            bird_velocity: 0.0,
            bird_x: 100.0,
            pipes: Vec::new(),
            score: 0,
            alive: true,
            frame_count: 0,
        }
    }
}

impl GameState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn get_observation(&self) -> Vec<f32> {
        let mut obs = vec![self.bird_y, self.bird_velocity];

        // Find the next pipe
        if let Some(next_pipe) = self.pipes.iter().find(|p| p.x + p.width > self.bird_x) {
            obs.push(next_pipe.x - self.bird_x);
            obs.push(next_pipe.gap_y);
            obs.push(next_pipe.gap_y + next_pipe.gap_height);
        } else {
            obs.push(800.0);
            obs.push(300.0);
            obs.push(500.0);
        }

        obs
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn flappybird_lib(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<GameState>()?;
    m.add_class::<Pipe>()?;
    Ok(())
}

pub const GRAVITY: f32 = 0.6;
pub const JUMP_STRENGTH: f32 = -10.0;
pub const BIRD_SIZE: f32 = 32.0;
pub const PIPE_WIDTH: f32 = 80.0;
pub const PIPE_GAP: f32 = 180.0;
pub const PIPE_SPEED: f32 = 3.0;
pub const SCREEN_WIDTH: f32 = 800.0;
pub const SCREEN_HEIGHT: f32 = 600.0;
