use ggez::event::{self, EventHandler};
use ggez::graphics::{self, Color, DrawMode, Mesh, Rect};
use ggez::input::keyboard::{KeyCode, KeyInput};
use ggez::{Context, GameResult};
use rand::Rng;
use std::time::{Duration, Instant};

use flappybird_lib::*;

const PIPE_SPAWN_MARGIN: f32 = 80.0; // Prevent impossible openings near floor/ceiling

struct FlappyBirdGame {
    state: GameState,
    last_pipe_spawn: Instant,
    pipe_spawn_interval: Duration,
}

impl FlappyBirdGame {
    fn new() -> Self {
        let mut game = FlappyBirdGame {
            state: GameState::new(),
            last_pipe_spawn: Instant::now(),
            pipe_spawn_interval: Duration::from_millis(1800),
        };
        game.spawn_pipe();
        game
    }

    fn spawn_pipe(&mut self) {
        let mut rng = rand::rng();
        let min_gap = PIPE_SPAWN_MARGIN;
        let max_gap = (SCREEN_HEIGHT - PIPE_GAP - PIPE_SPAWN_MARGIN).max(min_gap + 1.0);
        let gap_y = rng.random_range(min_gap..max_gap);

        self.state.pipes.push(Pipe {
            x: SCREEN_WIDTH,
            gap_y,
            gap_height: PIPE_GAP,
            width: PIPE_WIDTH,
            passed: false,
        });
    }

    fn update_physics(&mut self) {
        if !self.state.alive {
            return;
        }

        // Apply gravity
        self.state.bird_velocity += GRAVITY;
        self.state.bird_y += self.state.bird_velocity;

        // Clamp position so we do not drift outside the playfield
        if self.state.bird_y < 0.0 {
            self.state.bird_y = 0.0;
            self.state.alive = false;
            return;
        } else if self.state.bird_y > SCREEN_HEIGHT - BIRD_SIZE {
            self.state.bird_y = SCREEN_HEIGHT - BIRD_SIZE;
            self.state.alive = false;
            return;
        }

        // Update pipes
        let bird_x = self.state.bird_x;
        let bird_y = self.state.bird_y;

        for pipe in &mut self.state.pipes {
            pipe.x -= PIPE_SPEED;

            // Check if bird passed the pipe
            if !pipe.passed && pipe.x + pipe.width < bird_x {
                pipe.passed = true;
                self.state.score += 1;
            }

            // Check collision with pipe
            if Self::check_collision(bird_x, bird_y, pipe) {
                self.state.alive = false;
            }
        }

        // Remove off-screen pipes
        self.state.pipes.retain(|p| p.x + p.width > -10.0);

        // Spawn new pipes
        if self.last_pipe_spawn.elapsed() >= self.pipe_spawn_interval {
            self.spawn_pipe();
            self.last_pipe_spawn = Instant::now();
        }

        self.state.frame_count += 1;
    }

    fn check_collision(bird_x: f32, bird_y: f32, pipe: &Pipe) -> bool {
        let bird_rect = Rect::new(bird_x, bird_y, BIRD_SIZE, BIRD_SIZE);

        // Top pipe
        let top_pipe = Rect::new(pipe.x, 0.0, pipe.width, pipe.gap_y);
        // Bottom pipe
        let bottom_pipe = Rect::new(
            pipe.x,
            pipe.gap_y + pipe.gap_height,
            pipe.width,
            SCREEN_HEIGHT - pipe.gap_y - pipe.gap_height,
        );

        bird_rect.overlaps(&top_pipe) || bird_rect.overlaps(&bottom_pipe)
    }

    pub fn reset(&mut self) {
        self.state.reset();
        self.last_pipe_spawn = Instant::now();
        self.spawn_pipe();
    }
}

impl EventHandler for FlappyBirdGame {
    fn update(&mut self, _ctx: &mut Context) -> GameResult {
        self.update_physics();
        Ok(())
    }

    fn draw(&mut self, ctx: &mut Context) -> GameResult {
        let mut canvas = graphics::Canvas::from_frame(ctx, Color::from_rgb(135, 206, 235));

        // Draw bird
        let bird_rect = Rect::new(self.state.bird_x, self.state.bird_y, BIRD_SIZE, BIRD_SIZE);
        let bird_color = if self.state.alive {
            Color::from_rgb(255, 255, 0)
        } else {
            Color::from_rgb(255, 0, 0)
        };
        let bird_mesh = Mesh::new_rectangle(ctx, DrawMode::fill(), bird_rect, bird_color)?;
        canvas.draw(&bird_mesh, graphics::DrawParam::default());

        // Draw pipes
        for pipe in &self.state.pipes {
            // Top pipe
            let top_rect = Rect::new(pipe.x, 0.0, pipe.width, pipe.gap_y);
            let top_mesh =
                Mesh::new_rectangle(ctx, DrawMode::fill(), top_rect, Color::from_rgb(0, 200, 0))?;
            canvas.draw(&top_mesh, graphics::DrawParam::default());

            // Bottom pipe
            let bottom_rect = Rect::new(
                pipe.x,
                pipe.gap_y + pipe.gap_height,
                pipe.width,
                SCREEN_HEIGHT - pipe.gap_y - pipe.gap_height,
            );
            let bottom_mesh = Mesh::new_rectangle(
                ctx,
                DrawMode::fill(),
                bottom_rect,
                Color::from_rgb(0, 200, 0),
            )?;
            canvas.draw(&bottom_mesh, graphics::DrawParam::default());
        }

        // Draw score
        let score_text = graphics::Text::new(format!("Score: {}", self.state.score));
        canvas.draw(
            &score_text,
            graphics::DrawParam::default()
                .dest([10.0, 10.0])
                .color(Color::WHITE),
        );

        // Draw game over message
        if !self.state.alive {
            let game_over_text = graphics::Text::new("Game Over! Press R to restart");
            canvas.draw(
                &game_over_text,
                graphics::DrawParam::default()
                    .dest([SCREEN_WIDTH / 2.0 - 150.0, SCREEN_HEIGHT / 2.0])
                    .color(Color::RED),
            );
        }

        canvas.finish(ctx)?;
        Ok(())
    }

    fn key_down_event(&mut self, _ctx: &mut Context, input: KeyInput, _repeat: bool) -> GameResult {
        if let Some(keycode) = input.keycode {
            match keycode {
                KeyCode::Space => {
                    if self.state.alive {
                        self.state.bird_velocity = JUMP_STRENGTH;
                    }
                }
                KeyCode::R => {
                    self.reset();
                }
                _ => {}
            }
        }
        Ok(())
    }
}

fn main() -> GameResult {
    let cb = ggez::ContextBuilder::new("flappy_bird", "AI Training")
        .window_setup(ggez::conf::WindowSetup::default().title("Flappy Bird - AI Training"))
        .window_mode(ggez::conf::WindowMode::default().dimensions(SCREEN_WIDTH, SCREEN_HEIGHT));

    let (ctx, event_loop) = cb.build()?;
    let game = FlappyBirdGame::new();

    event::run(ctx, event_loop, game)
}
