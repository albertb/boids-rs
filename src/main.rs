use std::{cmp::Ordering, f32::consts::PI, ops::Range};

use bevy::{
    prelude::*,
    sprite::{MaterialMesh2dBundle, Mesh2dHandle},
    window::{PrimaryWindow, WindowResized},
};

use bevy_egui::{egui, EguiContexts, EguiPlugin};
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Exp};

#[derive(Resource)]
struct Parameters {
    window_width: f32,
    window_height: f32,
    number_of_boids: usize,
    view_distance: f32,

    cohesion_force: f32,
    separation_force: f32,
    separation_bias: f32,
    alignment_bias: f32,
    alignment_force: f32,
    steering_force: f32,

    fidelity: f32,

    min_speed: f32,
    max_speed: f32,

    bounce_off_walls: bool,
}

impl Default for Parameters {
    fn default() -> Self {
        Self {
            window_width: 100.0,
            window_height: 100.0,
            number_of_boids: 512,
            view_distance: 75.0,
            cohesion_force: 2.5,
            separation_force: 1.8,
            separation_bias: 1.1,
            alignment_force: 1.1,
            alignment_bias: 1.5,
            steering_force: 0.8,
            fidelity: 0.9,
            min_speed: 25.0,
            max_speed: 100.0,
            bounce_off_walls: false,
        }
    }
}

impl Parameters {
    // The valid range of x, y coordinates given the window size.
    fn window_x_range(&self) -> Range<f32> {
        -self.window_width / 2.0..self.window_width / 2.0
    }
    fn window_y_range(&self) -> Range<f32> {
        -self.window_height / 2.0..self.window_height / 2.0
    }

    // The maximum position vector given the window size.
    fn max_position(&self) -> Vec3 {
        Vec3::new(self.window_width / 2.0, self.window_width / 2.0, 0.)
    }
    // The minimum position vector given the window size.
    fn min_position(&self) -> Vec3 {
        Vec3::new(-self.window_height / 2.0, -self.window_height / 2.0, 0.)
    }
}

#[derive(Component, Debug)]
struct Boid {
    velocity: Vec2,
    weight: f32,
}

impl Boid {
    fn new(x: f32, y: f32, w: f32) -> Self {
        Self {
            velocity: Vec2::new(x, y),
            weight: w,
        }
    }
}

#[derive(Component, Default)]
struct Calculations {
    neighbours: i32,
    cohesion: Vec2,
    separation: Vec2,
    alignment: Vec2,
}

impl Calculations {
    fn reset(&mut self) {
        self.neighbours = 0;
        self.cohesion = Vec2::ZERO;
        self.separation = Vec2::ZERO;
        self.alignment = Vec2::ZERO;
    }
}

const BIRD_SIZE: f32 = 2.0;

fn setup(
    params: Res<Parameters>,
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
) {
    commands.spawn(Camera2dBundle::default());
    spawn_boids(params.number_of_boids, params, commands, meshes, materials);
}

fn spawn_boids(
    how_many: usize,
    params: Res<Parameters>,
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for i in 1..=how_many {
        let color = Color::hsl(360. * i as f32 / how_many as f32, 0.95, 0.7);
        let weight = 1.0 + Exp::new(20.0).unwrap().sample(&mut thread_rng()) * 10.0;
        let size = BIRD_SIZE * weight;

        commands.spawn((
            MaterialMesh2dBundle {
                mesh: Mesh2dHandle(meshes.add(Triangle2d::new(
                    Vec2::Y * size * 2.0,
                    Vec2::new(-size, -size),
                    Vec2::new(size, -size),
                ))),
                material: materials.add(color),
                transform: Transform::from_xyz(
                    thread_rng().gen_range(params.window_x_range()),
                    thread_rng().gen_range(params.window_y_range()),
                    0.,
                ),
                ..default()
            },
            Boid::new(
                thread_rng().gen_range(-params.max_speed..params.max_speed),
                thread_rng().gen_range(-params.max_speed..params.max_speed),
                weight,
            ),
            Calculations::default(),
        ));
    }
}

fn flock(params: Res<Parameters>, mut query: Query<(&Transform, &mut Calculations, &mut Boid)>) {
    let mut pairs = query.iter_combinations_mut();
    while let Some([(t1, mut c1, b1), (t2, mut c2, b2)]) = pairs.fetch_next() {
        if thread_rng().gen_range(0.0..=1.0) > params.fidelity {
            continue;
        }

        let distance = t1.translation.distance(t2.translation);
        if distance > params.view_distance {
            continue;
        }
        let distance = distance.max(0.001); // Avoid division by zero.
        let p1 = t1.translation.truncate();
        let p2 = t2.translation.truncate();

        // Seperation should be stronger for boids closer to each other.
        let separation_factor = 1.0 / distance.powf(params.separation_bias);

        // Cosine similarity between the two velocities: 1.0 if same, -1.0 if opposite.
        let similarity =
            b1.velocity.dot(b2.velocity) / (b1.velocity.length() * b2.velocity.length());
        // When bias > 1, prefers boids already going in a similar drection.
        // When bias < 1, prefers boids going in the opposite direction.
        let bias = params.alignment_bias;
        let alignment_factor = bias.powf(similarity) / if bias > 1.0 { bias } else { 1.0 / bias };

        // Larger boids have a stronger influence.
        let b1w = b1.weight.powi(2) / b2.weight.powi(2);
        let b2w = b2.weight.powi(2) / b1.weight.powi(2);

        c1.neighbours += 1;
        c1.cohesion += p2 * b2w;
        c1.separation += (p1 - p2) * separation_factor * b2w;
        c1.alignment += b2.velocity * alignment_factor * b2w;

        c2.neighbours += 1;
        c2.cohesion += p1 * b1w;
        c2.separation += (p2 - p1) * separation_factor * b1w;
        c2.alignment += b1.velocity * alignment_factor * b1w;
    }

    for (_, mut c, mut b) in &mut query {
        if c.neighbours <= 0 {
            continue;
        }

        let cohesion = -(c.cohesion / c.neighbours as f32).clamp_length_max(params.steering_force);
        let separation = c.separation.clamp_length_max(params.steering_force);
        let alignment = c.alignment.clamp_length_max(params.steering_force);

        b.velocity = b.velocity
            + params.cohesion_force * cohesion
            + params.separation_force * separation
            + params.alignment_force * alignment;
        b.velocity = b.velocity.clamp_length(params.min_speed, params.max_speed);
        c.reset(); // Reset calculations for next frame.
    }
}

fn adjust_number_of_boids(
    mut commands: Commands,
    meshes: ResMut<Assets<Mesh>>,
    materials: ResMut<Assets<ColorMaterial>>,
    params: Res<Parameters>,
    query: Query<Entity, With<Boid>>,
) {
    let count = query.iter().count();
    match count.cmp(&params.number_of_boids) {
        Ordering::Less => spawn_boids(
            params.number_of_boids - count,
            params,
            commands,
            meshes,
            materials,
        ),
        Ordering::Greater => {
            for (i, e) in query.iter().enumerate() {
                if i >= params.number_of_boids {
                    commands.entity(e).despawn();
                }
            }
        }
        _ => (),
    };
}

fn handle_walls(params: Res<Parameters>, mut query: Query<(&mut Transform, &mut Boid)>) {
    for (mut t, mut b) in &mut query {
        let x = t.translation.x;
        if !params.window_x_range().contains(&x) && b.velocity.x.signum() == x.signum() {
            if params.bounce_off_walls {
                b.velocity.x *= -1.0;
            } else {
                t.translation.x *= -1.0;
            }
        }
        let y = t.translation.y;
        if !params.window_y_range().contains(&y) && b.velocity.y.signum() == y.signum() {
            if params.bounce_off_walls {
                b.velocity.y *= -1.0;
            } else {
                t.translation.y *= -1.0;
            }
        }
    }
}

fn handle_mouse(
    window: Query<&Window, With<PrimaryWindow>>,
    camera: Query<(&Camera, &GlobalTransform)>,
    buttons: Res<ButtonInput<MouseButton>>,
    params: Res<Parameters>,
    mut query: Query<(&Transform, &mut Boid)>,
) {
    // Follow or avoid the mouse pointer.
    let (camera, camera_transform) = camera.single();
    if let Some(mouse_position) = window
        .single()
        .cursor_position()
        .and_then(|cursor| camera.viewport_to_world(camera_transform, cursor))
        .map(|ray| ray.origin)
    {
        // Left click attracts, right click repels.
        let direction = match buttons.get_pressed().last() {
            Some(MouseButton::Left) => 1.0,
            Some(MouseButton::Right) => -1.0,
            _ => return, // No effect when neither button is pressed.
        };

        let mouse_position = mouse_position.truncate();
        for (t, mut boid) in &mut query {
            let position = t.translation.truncate();
            let distance = position.distance(mouse_position);

            // Allow the mouse to affect boids further away.
            if distance > params.view_distance * 4.0 {
                continue;
            }
            let target = (mouse_position - position) * direction;

            boid.velocity = (boid.velocity
                + target * params.steering_force * params.cohesion_force * 0.5)
                .clamp_length_max(params.max_speed);
        }
    }
}

fn fly(
    time: Res<Time>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut query: Query<(&mut Transform, &Handle<ColorMaterial>, &Boid)>,
) {
    // Slow-motion
    // if time.elapsed().as_millis() % 500 > 15 {
    //     return;
    // }

    for (mut transform, material_handle, boid) in &mut query {
        let direction = (transform.rotation * Vec3::Y).truncate();
        let target = boid.velocity.normalize();

        // Rotate boid towards its velocity vector.
        let target_rotation = Quat::from_rotation_arc_2d(direction, target);
        transform.rotate(target_rotation);

        // Color the boid based on its velocity angle.
        if let Some(material) = materials.get_mut(material_handle) {
            material.color = Color::hsl(
                360. * (target.angle_between(Vec2::Y) + PI) / (2.0 * PI),
                0.95,
                0.7,
            );
        }

        transform.translation.x += boid.velocity.x * time.delta_seconds();
        transform.translation.y += boid.velocity.y * time.delta_seconds();
    }
}

fn parameters_ui(
    mut contexts: EguiContexts,
    mut params: ResMut<Parameters>,
    mut boids: Query<&mut Transform, With<Boid>>,
) {
    egui::Window::new("Parameters")
        .default_open(false)
        .show(contexts.ctx_mut(), |ui| {
            ui.add(
                egui::Slider::new(&mut params.number_of_boids, 8..=2048)
                    .logarithmic(true)
                    .text("Number of boids"),
            )
            .on_hover_text("Too many boids will affect frame rate.");
            ui.separator();
            ui.add(egui::Slider::new(&mut params.view_distance, 0.0..=500.0).text("View distance"))
                .on_hover_text("How far away each boid can see.");
            ui.add(
                egui::Slider::new(&mut params.cohesion_force, 0.0..=100.0)
                    .logarithmic(true)
                    .text("Cohesion force"),
            )
            .on_hover_text("How strongly to aim towards other boids.");
            ui.add(
                egui::Slider::new(&mut params.separation_force, 0.0..=100.0)
                    .logarithmic(true)
                    .text("Separation force"),
            )
            .on_hover_text("How strongly to aim away from close boids.");
        ui.add(
            egui::Slider::new(&mut params.separation_bias, 0.01..=10.0)
                .logarithmic(true)
                .text("Separation bias"),
        )
        .on_hover_text("How strongly should the separation force be affected by distance. Larger values means closer boids have a larger influence.");
            ui.add(
                egui::Slider::new(&mut params.alignment_force, 0.0..=100.0)
                    .logarithmic(true)
                    .text("Alignment force"),
            )
            .on_hover_text("How strongly to align with nearby boids.");
            ui.add(
                egui::Slider::new(&mut params.alignment_bias, 0.01..=100.0)
                    .logarithmic(true)
                    .text("Alignment bias"),
            ).on_hover_text("Whether to align with boids going in a similar direction. A negative value here means to align with boids going in the opposite direction.");

            ui.add(
                egui::Slider::new(&mut params.steering_force, 0.0..=100.0)
                    .logarithmic(true)
                    .text("Steering force"),
            ).on_hover_text("How strongly to steer when changing direction.");
            ui.separator();
            ui.add(egui::Slider::new(&mut params.fidelity, 0.01..=1.0).text("Fidelity")).on_hover_text("How often should boids steer at all.");
            ui.separator();
            let max_speed = params.max_speed;
            ui.add(
                egui::Slider::new(&mut params.min_speed, 10.0..=max_speed).text("Minimum speed"),
            );
            let min_speed = params.min_speed;
            ui.add(
                egui::Slider::new(&mut params.max_speed, min_speed..=500.0).text("Maximum speed"),
            );
            ui.add(egui::Checkbox::new(
                &mut params.bounce_off_walls,
                "Bounce off walls",
            ));
            ui.separator();
            if ui.button("Restart").clicked() {
                for mut t in &mut boids {
                    t.translation.x = thread_rng()
                        .gen_range(params.window_x_range());
                    t.translation.y = thread_rng()
                        .gen_range(params.window_y_range());
                }
            }
        });
}

fn window_resize(
    mut resize_reader: EventReader<WindowResized>,
    mut params: ResMut<Parameters>,
    mut query: Query<&mut Transform, With<Boid>>,
) {
    if let Some(e) = resize_reader.read().last() {
        if params.window_width == e.width && params.window_height == e.height {
            return;
        }
        params.window_width = e.width;
        params.window_height = e.height;

        for mut t in &mut query {
            // Constrain the boids to the new window size.
            t.translation = t
                .translation
                .clamp(params.min_position(), params.max_position());
        }
    }
}

fn main() {
    let mut app = App::new();
    app.add_plugins((
        DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                resolution: (640., 480.).into(),
                ..default()
            }),
            ..default()
        }),
        EguiPlugin,
    ))
    .insert_resource(Parameters::default())
    .add_systems(Startup, setup)
    .add_systems(
        Update,
        (
            parameters_ui,
            adjust_number_of_boids,
            (flock, handle_mouse, handle_walls, fly).chain(),
        ),
    )
    .add_systems(PostUpdate, window_resize);

    #[cfg(debug_assertions)]
    {
        use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};
        // Log the FPS in debug builds.
        app.add_plugins((FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin::default()));
    }

    app.run();
}
