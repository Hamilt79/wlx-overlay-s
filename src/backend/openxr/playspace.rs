use glam::{Affine3A, Quat, Vec3A};
use libmonado::{Monado, Pose, ReferenceSpaceType};

use crate::{
    backend::{common::OverlayContainer, input::InputState},
    state::AppState,
};

use super::overlay::OpenXrOverlayData;

struct MoverData<T> {
    pose: Affine3A,
    hand: usize,
    hand_pose: T,
    velocity: Vec3A,
}

pub(super) struct PlayspaceMover {
    last_transform: Affine3A,
    floor_offset: f32,
    space_fling_enabled: bool,
    momentum_velocity: Vec3A,
    drag: Option<MoverData<Vec3A>>,
    rotate: Option<MoverData<Quat>>,
}

impl PlayspaceMover {
    pub fn new(monado: &mut Monado) -> anyhow::Result<Self> {
        log::info!("Monado: using space offset API");

        let Ok(stage) = monado.get_reference_space_offset(ReferenceSpaceType::Stage) else {
            anyhow::bail!("Space offsets not supported.");
        };

        log::debug!("STAGE is at {:?}, {:?}", stage.position, stage.orientation);

        // initial offset
        let last_transform =
            Affine3A::from_rotation_translation(stage.orientation.into(), stage.position.into());

        Ok(Self {
            last_transform,
            floor_offset: 0.0,
            space_fling_enabled: false,
            momentum_velocity: Vec3A::ZERO,
            drag: None,
            rotate: None,
        })
    }

    pub fn update(
        &mut self,
        overlays: &mut OverlayContainer<OpenXrOverlayData>,
        state: &AppState,
        monado: &mut Monado,
    ) {
        for pointer in &state.input_state.pointers {
            if pointer.now.space_reset {
                if !pointer.before.space_reset {
                    log::info!("Space reset");
                    self.reset_offset(monado);
                }
                return;
            }
        }

        if let Some(mut data) = self.rotate.take() {
            let pointer = &state.input_state.pointers[data.hand];
            if !pointer.now.space_rotate {
                self.last_transform = data.pose;
                log::info!("End space rotate");
                return;
            }

            let new_hand =
                Quat::from_affine3(&(data.pose * state.input_state.pointers[data.hand].raw_pose));

            let dq = new_hand * data.hand_pose.conjugate();
            let mut space_transform = if state.session.config.space_rotate_unlocked {
                Affine3A::from_quat(dq)
            } else {
                let rel_y = f32::atan2(
                    2.0 * dq.y.mul_add(dq.w, dq.x * dq.z),
                    2.0f32.mul_add(dq.w.mul_add(dq.w, dq.x * dq.x), -1.0),
                );

                Affine3A::from_rotation_y(rel_y)
            };
            let offset = (space_transform.transform_vector3a(state.input_state.hmd.translation)
                - state.input_state.hmd.translation)
                * -1.0;

            space_transform.translation = offset;

            data.pose *= space_transform;
            data.hand_pose = new_hand;

            self.apply_offset(data.pose, monado);
            self.rotate = Some(data);
        } else {
            for (i, pointer) in state.input_state.pointers.iter().enumerate() {
                if pointer.now.space_rotate {
                    let hand_pose = Quat::from_affine3(&(self.last_transform * pointer.raw_pose));
                    self.rotate = Some(MoverData {
                        pose: self.last_transform,
                        hand: i,
                        hand_pose,
                        velocity: Vec3A::ZERO,
                    });
                    self.drag = None;
                    log::info!("Start space rotate");
                    return;
                }
            }
        }

        if let Some(mut data) = self.drag.take() {
            let pointer = &state.input_state.pointers[data.hand];
            if !pointer.now.space_drag {
                self.momentum_velocity = data.velocity;
                self.last_transform = data.pose;
                log::info!("End space drag");
                return;
            }

            let new_hand = data
                .pose
                .transform_point3a(state.input_state.pointers[data.hand].raw_pose.translation);
            let relative_pos =
                (new_hand - data.hand_pose) * state.session.config.space_drag_multiplier;
            data.velocity = relative_pos;
            if relative_pos.length_squared() > 1000.0 {
                log::warn!("Space drag too fast, ignoring");
                return;
            }

            let overlay_offset = data.pose.inverse().transform_vector3a(relative_pos) * -1.0;

            overlays.iter_mut().for_each(|overlay| {
                if overlay.state.grabbable {
                    overlay.state.dirty = true;
                    overlay.state.transform.translation += overlay_offset;
                }
            });

            data.pose.translation += relative_pos;
            data.hand_pose = new_hand;

            self.apply_offset(data.pose, monado);
            self.drag = Some(data);
        } else {
            for (i, pointer) in state.input_state.pointers.iter().enumerate() {
                if pointer.now.space_drag {
                    let hand_pos = self
                        .last_transform
                        .transform_point3a(pointer.raw_pose.translation);
                    self.drag = Some(MoverData {
                        pose: self.last_transform,
                        hand: i,
                        hand_pose: hand_pos,
                        velocity: Vec3A::ZERO,
                    });
                    log::info!("Start space drag");
                    return;
                }
            }
        }

        state
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_fling && !p.before.space_fling)
            .then(|| self.space_fling_enabled ^= true);

        // const FLING_STRENGTH: f32 = 2.0;
        const CONSIDER_FLOOR: bool = false;

        let user_is_interacting = state
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_drag || p.now.space_rotate);

        if !user_is_interacting && self.space_fling_enabled {

            let mut new_pose = self.last_transform;
            new_pose.translation += self.momentum_velocity * state.session.config.space_fling_multiplier;

            if CONSIDER_FLOOR && (new_pose.translation.y > 0.0) {
                new_pose.translation.y = 0.0;
                self.momentum_velocity = Vec3A::ZERO;
            }

            if new_pose.translation != self.last_transform.translation {
                self.last_transform = new_pose;
                self.apply_offset(new_pose, monado);
            }
        } else {
            self.momentum_velocity = Vec3A::ZERO;
        }
    }

    pub fn reset_offset(&mut self, monado: &mut Monado) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by manual reset");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by manual reset");
            self.rotate = None;
        }

        self.last_transform = Affine3A::IDENTITY;
        self.apply_offset(self.last_transform, monado);
    }

    pub fn fix_floor(&mut self, input: &InputState, monado: &mut Monado) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by fix floor");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by fix floor");
            self.rotate = None;
        }

        let y1 = input.pointers[0].raw_pose.translation.y;
        let y2 = input.pointers[1].raw_pose.translation.y;
        let delta = y1.min(y2) - 0.03;
        self.floor_offset = delta;
        self.apply_offset(self.last_transform, monado);
    }

    fn apply_offset(&self, transform: Affine3A, monado: &mut Monado) {
        let pose = Pose {
            position: (transform.translation + Vec3A::new(0.0, self.floor_offset, 0.0)).into(),
            orientation: Quat::from_affine3(&transform).into(),
        };
        let _ = monado.set_reference_space_offset(ReferenceSpaceType::Stage, pose);
    }
}
