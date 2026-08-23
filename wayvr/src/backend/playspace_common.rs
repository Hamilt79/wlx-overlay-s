use glam::{Affine3A, Quat, Vec2, Vec3A};
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
    /// true while gravity is actively moving the playspace
    active: bool,
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
            active: false,
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
        self.active = false;
    }

    pub fn update(&mut self, par: SpaceGravityUpdateParams) -> Option<SpaceGravityUpdateResult> {
        if par.dragging || !par.config.space_gravity_enabled {
            self.active = false;
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
            self.active = true;
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

        // velocity fell below the threshold: gravity has come to rest
        self.active = false;

        None
    }

    /// true while gravity is moving the playspace
    pub const fn is_active(&self) -> bool {
        self.active
    }
}

pub struct SpaceBoostUpdateParams<'a> {
    pub dt: f32,
    pub config: &'a GeneralConfig,
    /// HMD pose expressed in the same frame as the stage offset translation, i.e. the
    /// current offset multiplied by the runtime-reported HMD pose
    pub hmd: Affine3A,
    /// stick deflection: x strafes, y drives forward
    pub stick: Vec2,
}

/// Analog stick locomotion, toggled on and off by the `space_boost` binding.
///
/// The translation it contributes is tracked separately from the stage offset so that a
/// plain space reset can put it back, while a full reset can throw it away.
pub struct SpaceBoost {
    enabled: bool,
    /// total translation this has contributed to the stage offset
    offset: Vec3A,
}

impl SpaceBoost {
    pub const fn new() -> Self {
        Self {
            enabled: false,
            offset: Vec3A::ZERO,
        }
    }

    pub const fn enabled(&self) -> bool {
        self.enabled
    }

    pub const fn offset(&self) -> Vec3A {
        self.offset
    }

    /// flips the mode and reports the new state
    pub fn toggle(&mut self) -> bool {
        self.enabled = !self.enabled;
        self.enabled
    }

    /// forgets the accumulated translation and hands it back, so the caller can decide
    /// whether to subtract it from the stage offset (snap back) or leave it (stay put)
    pub fn take_offset(&mut self) -> Vec3A {
        let offset = self.offset;
        self.offset = Vec3A::ZERO;
        offset
    }

    /// Returns the translation to add to the stage offset this frame, or None when the
    /// mode is off or the stick is inside the deadzone.
    ///
    /// Note the sign: moving the stage forward carries the world forward with it, so the
    /// user travels the opposite way. The returned delta is already negated for that.
    pub fn update(&mut self, par: &SpaceBoostUpdateParams) -> Option<Vec3A> {
        if !self.enabled {
            return None;
        }

        let deflection = par.stick.length();
        let deadzone = par.config.space_boost_deadzone.clamp(0.0, 0.99);
        if deflection <= deadzone {
            return None;
        }

        // remap past the deadzone so speed ramps from a standstill at its edge rather
        // than jumping straight to a fraction of full speed
        let throttle = ((deflection - deadzone) / (1.0 - deadzone)).clamp(0.0, 1.0);

        // flatten the HMD forward onto the horizontal plane: looking up or down must not
        // send the playspace vertically
        let forward = par.hmd.transform_vector3a(Vec3A::NEG_Z);
        let forward = Vec3A::new(forward.x, 0.0, forward.z).normalize_or_zero();
        if forward == Vec3A::ZERO {
            // looking straight up or down leaves no usable heading
            return None;
        }
        let right = Vec3A::new(-forward.z, 0.0, forward.x);

        let heading = (right * par.stick.x + forward * par.stick.y).normalize_or_zero();
        if heading == Vec3A::ZERO {
            return None;
        }

        let delta = heading * (-throttle * par.config.space_boost_speed * par.dt);
        self.offset += delta;
        Some(delta)
    }
}
