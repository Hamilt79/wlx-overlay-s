use glam::{Affine3A, Quat, Vec3A};
use wlx_common::config::GeneralConfig;

use crate::windowing::manager::OverlayWindowManager;

pub struct SpaceGravityUpdateParams<'a> {
    pub dt: f32,
    pub dragging: bool,
    pub config: &'a GeneralConfig,
    pub floor_height: f32,
}

pub struct SpaceGravity {
    velocity: Vec3A,
    space_pos: Vec3A,
    /// stage rotation captured when the drag ended; gravity only moves the playspace
    space_rot: Quat,
}

pub fn shift_world<OverlayData>(
    overlays: &mut OverlayWindowManager<OverlayData>,
    anchor: &mut Affine3A,
    before: &Affine3A,
    after: &Affine3A,
) {
    let correction = after.inverse() * before;
    *anchor = correction * *anchor;

    overlays.values_mut().for_each(|overlay| {
        let Some(state) = overlay.config.active_state.as_mut() else {
            return;
        };
        state.transform = correction * state.transform;
        overlay.config.dirty = true;
    });
}

pub struct SpaceGravityUpdateResult {
    pub playspace_pose: Affine3A,
    /// pose applied on the previous update() call
    pub previous_pose: Affine3A,
}

impl SpaceGravity {
    pub fn new() -> Self {
        Self {
            velocity: Vec3A::default(),
            space_pos: Vec3A::default(),
            space_rot: Quat::IDENTITY,
        }
    }

    pub fn mark_end_drag(
        &mut self,
        config: &GeneralConfig,
        hand_pos_diff: Vec3A,
        space_pose: Affine3A,
        dt: f32,
    ) {
        if config.space_gravity_enabled {
            self.velocity = hand_pos_diff * config.space_gravity_fling_strength / dt;
            self.space_pos = space_pose.translation;
            self.space_rot = Quat::from_affine3(&space_pose);
        } else {
            self.reset();
        }
    }

    pub fn reset(&mut self) {
        self.velocity = Vec3A::default();
        self.space_pos = Vec3A::default();
        self.space_rot = Quat::IDENTITY;
    }

    pub fn update(&mut self, par: SpaceGravityUpdateParams) -> Option<SpaceGravityUpdateResult> {
        if par.dragging || !par.config.space_gravity_enabled {
            return None;
        }

        let prev_pos = self.space_pos;

        self.velocity.y += par.config.space_gravity_gravity * par.dt;

        // terminal velocity
        self.velocity.y = self.velocity.y.min(200.0);

        self.velocity *= (par.config.space_gravity_damping).powf(par.dt * 10.0);

        self.space_pos += self.velocity * par.dt;

        self.space_pos.y = self.space_pos.y.min(par.floor_height);

        if self.space_pos.y >= par.floor_height
        /* at floor height or below */
        {
            // apply ground friction
            self.velocity *= 1.0 - par.config.space_gravity_ground_friction * par.dt * 10.0;
        }

        if self.velocity.length_squared() > 0.00003 {
            // Space position changed. Gravity only translates the playspace, so the
            // rotation captured at the end of the drag is carried through unchanged --
            // rebuilding the pose from translation alone would snap any space rotation
            // (or the yaw applied by a recenter) back to identity.
            return Some(SpaceGravityUpdateResult {
                playspace_pose: Affine3A::from_rotation_translation(
                    self.space_rot,
                    self.space_pos.into(),
                ),
                previous_pose: Affine3A::from_rotation_translation(self.space_rot, prev_pos.into()),
            });
        }

        None
    }
}
