use glam::{Affine3A, Quat, Vec2, Vec3, Vec3A, vec3a};
use libmonado::{Monado, Pose, ReferenceSpaceType};
use wgui::log::LogErr;

use crate::{
    backend::{
        playspace_common::{
            self, SpaceBoost, SpaceBoostUpdateParams, SpaceGravity, SpaceGravityUpdateParams,
        },
        task::PlayspaceTask,
    },
    state::{AppState, PlayspaceState, load_playspace_state, save_playspace_state},
    windowing::manager::OverlayWindowManager,
};

use super::overlay::OpenXrOverlayData;

struct MoverData<T> {
    pose: Affine3A,
    hand: usize,
    hand_pose: T,
}

/// Space rotate is recomputed absolutely from the grab pose each frame, so it needs
/// more state than the incremental `MoverData`.
struct RotateData {
    /// stage offset at the moment the gesture started
    initial_pose: Affine3A,
    /// stage offset currently applied to the runtime
    pending_pose: Affine3A,
    /// stage offset applied on the previous frame, for `shift_world`
    previous_pose: Affine3A,
    hand: usize,
    /// hand orientation in the fixed world frame at the moment the gesture started
    initial_hand_pose: Quat,
    /// world-frame point the rotation pivots around, fixed for the whole gesture
    initial_pivot: Vec3A,
}

pub(super) struct PlayspaceMover {
    drag: Option<MoverData<Vec3A>>,
    rotate: Option<RotateData>,
    gravity: SpaceGravity,
    boost: SpaceBoost,
    /// cached stage offset for the boost path. None means it has gone stale and must be
    /// re-read from the runtime before the next boost step
    boost_base: Option<Affine3A>,
    playspace_state: PlayspaceState,
}

impl PlayspaceMover {
    pub fn new() -> Self {
        log::info!("Monado: using space offset API");

        Self {
            drag: None,
            rotate: None,
            gravity: SpaceGravity::new(),
            boost: SpaceBoost::new(),
            boost_base: None,
            playspace_state: load_playspace_state().unwrap_or_default(),
        }
    }

    pub fn handle_task(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
        task: PlayspaceTask,
    ) {
        let Some(monado) = &mut app.monado_state else {
            return; // monado not available
        };

        match task {
            PlayspaceTask::FixFloor => {
                self.fix_floor(app, overlays);
            }
            PlayspaceTask::Reset => {
                self.reset_offset(app, overlays);
            }
            PlayspaceTask::FullReset => {
                self.full_reset(app, overlays);
            }
            PlayspaceTask::Recenter => {
                self.recenter(app, overlays);
            }
            PlayspaceTask::SaveCenter => {
                self.save_center(&mut monado.ipc);
            }
            PlayspaceTask::ResetCenter => {
                self.playspace_state.openxr_space_center = Affine3A::IDENTITY;
            }
        }
    }

    pub fn update(
        &mut self,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
        app: &mut AppState,
    ) {
        let Some(monado) = &mut app.monado_state else {
            return; // monado not available
        };

        // `space_fling` toggles space gravity on and off from a controller binding, so the
        // user does not have to open the dashboard to stop drifting. Session-only: the
        // value is not written back to the config file.
        if app
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_fling && !p.before.space_fling)
        {
            let enabled = !app.session.config.space_gravity_enabled;
            app.session.config.space_gravity_enabled = enabled;
            if !enabled {
                // drop any in-flight velocity so re-enabling does not resume the old fling
                self.gravity.reset();
            }
            log::info!("Space gravity {}", if enabled { "enabled" } else { "disabled" });
        }

        // `space_boost` toggles stick locomotion. The distance already travelled is kept
        // when switching off, so a plain space reset still returns you to it.
        if app
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_boost && !p.before.space_boost)
        {
            let enabled = self.boost.toggle();
            log::info!("Space boost {}", if enabled { "enabled" } else { "disabled" });
        }

        for pointer in &app.input_state.pointers {
            if pointer.now.space_reset {
                if !pointer.before.space_reset {
                    log::info!("Space reset");
                    self.reset_offset(app, overlays);
                }
                return;
            }
        }

        if let Some(mut data) = self.rotate.take() {
            let pointer = &app.input_state.pointers[data.hand];
            if !pointer.now.space_rotate {
                log::info!("End space rotate");
                return;
            }

            // `raw_pose` is reported relative to the (already offset) stage space, so
            // multiplying it by the current offset yields the hand pose in the fixed world
            // frame. `delta` therefore depends only on physical hand movement, which lets
            // us recompute the rotation absolutely from the grab pose on every frame
            // instead of accumulating per-frame deltas.
            let new_hand = Quat::from_affine3(&(data.pending_pose * pointer.raw_pose)).normalize();
            let mut delta = (new_hand * data.initial_hand_pose.conjugate()).normalize();
            if delta.w < 0.0 {
                // take the shortest path around the rotation
                delta = -delta;
            }

            let mut space_transform = if app.session.config.space_rotate_unlocked {
                Affine3A::from_quat(delta)
            } else {
                let rel_y = f32::atan2(
                    2.0 * delta.y.mul_add(delta.w, delta.x * delta.z),
                    2.0f32.mul_add(delta.w.mul_add(delta.w, delta.x * delta.x), -1.0),
                );

                Affine3A::from_rotation_y(rel_y)
            };

            // pivot around the HMD position captured at grab time, so the pivot does not
            // slide out from under the gesture when the user moves their head
            space_transform.translation =
                data.initial_pivot - space_transform.transform_point3a(data.initial_pivot);

            data.pending_pose = space_transform * data.initial_pose;

            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(
                    overlays,
                    &mut app.anchor,
                    &data.previous_pose,
                    &data.pending_pose,
                );
            }
            data.previous_pose = data.pending_pose;

            apply_offset(data.pending_pose, &mut monado.ipc);
            self.rotate = Some(data);
        } else {
            // start space rotate
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_rotate {
                    let Ok(pose) = monado
                        .ipc
                        .get_reference_space_offset(ReferenceSpaceType::Stage)
                        .log_err("Could not get reference space offset.")
                        .map(|p| {
                            Affine3A::from_rotation_translation(
                                p.orientation.into(),
                                p.position.into(),
                            )
                        })
                    else {
                        return;
                    };

                    let initial_hand_pose =
                        Quat::from_affine3(&(pose * pointer.raw_pose)).normalize();
                    let initial_pivot = pose.transform_point3a(app.input_state.hmd.translation);

                    self.rotate = Some(RotateData {
                        initial_pose: pose,
                        pending_pose: pose,
                        previous_pose: pose,
                        hand: i,
                        initial_hand_pose,
                        initial_pivot,
                    });
                    self.drag = None;
                    log::info!("Start space rotate");
                    return;
                }
            }
        }

        if let Some(mut data) = self.drag.take() {
            let new_hand = data
                .pose
                .transform_point3a(app.input_state.pointers[data.hand].raw_pose.translation);
            let relative_pos = if app.session.config.space_drag_unlocked {
                new_hand - data.hand_pose
            } else {
                vec3a(0., new_hand.y - data.hand_pose.y, 0.)
            } * app.session.config.space_drag_multiplier;
            let pointer = &app.input_state.pointers[data.hand];

            if !pointer.now.space_drag {
                self.gravity.mark_end_drag(
                    &app.session.config,
                    relative_pos,
                    data.pose,
                    app.delta_time,
                );

                log::info!("End space drag");
                return;
            }

            if relative_pos.length_squared() > 1000.0 {
                log::warn!("Space drag too fast, ignoring");
                return;
            }

            let before_pose = data.pose;
            data.pose.translation += relative_pos;
            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(overlays, &mut app.anchor, &before_pose, &data.pose);
            }
            data.hand_pose = new_hand;

            apply_offset(data.pose, &mut monado.ipc);
            self.drag = Some(data);
        } else {
            // start space drag
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_drag {
                    let Ok(pose) = monado
                        .ipc
                        .get_reference_space_offset(ReferenceSpaceType::Stage)
                        .log_err("Could not get reference space offset.")
                        .map(|p| {
                            Affine3A::from_rotation_translation(
                                p.orientation.into(),
                                p.position.into(),
                            )
                        })
                    else {
                        return;
                    };

                    let hand_pos = pose.transform_point3a(pointer.raw_pose.translation);
                    self.drag = Some(MoverData {
                        pose,
                        hand: i,
                        hand_pose: hand_pos,
                    });
                    log::info!("Start space drag");
                    return;
                }
            }
        }

        if let Some(res) = self.gravity.update(SpaceGravityUpdateParams {
            dt: app.delta_time,
            dragging: self.drag.is_some(),
            config: &app.session.config,
            floor_height: app.session.config.space_gravity_floor_height,
        }) {
            apply_offset(res.playspace_pose, &mut monado.ipc);

            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(
                    overlays,
                    &mut app.anchor,
                    &res.previous_pose,
                    &res.playspace_pose,
                );
            }
        }

        if self.gravity.just_landed() {
            // landing discards the boost translation and pulls the user back
            self.clear_boost(app, overlays);
        }

        if self.drag.is_none() && self.rotate.is_none() && !self.gravity.is_active() {
            self.update_boost(app, overlays);
        } else {
            // something else moved the stage: the cached base is no longer valid
            self.boost_base = None;
        }
    }

    pub fn recenter(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        let Some(monado) = &mut app.monado_state else {
            return;
        };

        if self.drag.is_some() {
            log::info!("Space drag interrupted by recenter");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by recenter");
            self.rotate = None;
        }

        // recenter keeps the boost translation but moves the stage, so the cache is stale
        self.boost_base = None;

        let input = &app.input_state;
        let anchor = &mut app.anchor;

        let Ok(mut pose) = monado
            .ipc
            .get_reference_space_offset(ReferenceSpaceType::Stage)
            .inspect_err(|e| log::warn!("Could not recenter due to libmonado error: {e:?}"))
        else {
            return;
        };

        let before =
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());

        let cur_rot: Quat = pose.orientation.into();
        let cur_pos: Vec3 = pose.position.into();

        let mut stage_offset = Affine3A::from_rotation_translation(cur_rot, cur_pos);

        let horiz_hmd_pos = Vec3::new(input.hmd.translation.x, 0.0, input.hmd.translation.z);

        let fwd = input.hmd.transform_vector3a(Vec3A::NEG_Z);
        let horiz_len_sq = fwd.x.mul_add(fwd.x, fwd.z * fwd.z);

        let hmd_yaw = if horiz_len_sq > f32::EPSILON {
            let yaw = (-fwd.x).atan2(-fwd.z);
            Quat::from_rotation_y(yaw)
        } else {
            Quat::IDENTITY
        };

        let recenter_offset = Affine3A::from_rotation_translation(hmd_yaw, horiz_hmd_pos);

        stage_offset *= recenter_offset;

        let (_, new_rot, new_pos) = stage_offset.to_scale_rotation_translation();

        pose.position = new_pos.into();

        if horiz_len_sq > f32::EPSILON {
            pose.orientation = new_rot.into();
        }

        let _ = monado
            .ipc
            .set_reference_space_offset(ReferenceSpaceType::Stage, pose)
            .inspect_err(|e| log::warn!("Could not recenter due to libmonado error: {e:?}"));

        self.gravity.reset();

        if !app.session.config.space_drag_affects_world {
            let after =
                Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());
            playspace_common::shift_world(overlays, anchor, &before, &after);
        }
    }

    pub fn reset_offset(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        let Some(monado) = &mut app.monado_state else {
            return;
        };

        if self.drag.is_some() {
            log::info!("Space drag interrupted by manual reset");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by manual reset");
            self.rotate = None;
        }

        let Ok(pose) = monado
            .ipc
            .get_reference_space_offset(ReferenceSpaceType::Stage)
            .inspect_err(|e| log::warn!("Could not reset offset due to libmonado error: {e:?}"))
        else {
            return;
        };

        let before =
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());

        self.gravity.reset();
        // the boost translation deliberately survives a plain reset; only a full reset
        // (or fix floor / a gravity landing) throws it away
        let mut offset = self.playspace_state.openxr_space_center;
        offset.translation += self.boost.offset();
        self.boost_base = Some(offset);
        apply_offset(offset, &mut monado.ipc);

        if !app.session.config.space_drag_affects_world {
            let after = offset;
            playspace_common::shift_world(overlays, &mut app.anchor, &before, &after);
        }
    }

    pub fn fix_floor(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        let Some(monado) = &mut app.monado_state else {
            return;
        };

        if self.drag.is_some() {
            log::info!("Space drag interrupted by fix floor");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by fix floor");
            self.rotate = None;
        }

        let input = &app.input_state;
        let anchor = &mut app.anchor;

        let Ok(mut pose) = monado
            .ipc
            .get_reference_space_offset(ReferenceSpaceType::Stage)
            .inspect_err(|e| log::warn!("Could not fix floor due to libmonado error: {e:?}"))
        else {
            return;
        };

        let before =
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());

        let y1 = input.pointers[0].raw_pose.translation.y;
        let y2 = input.pointers[1].raw_pose.translation.y;
        let delta = y1.min(y2) - 0.05;

        pose.position.y += delta;

        self.playspace_state.openxr_space_center.translation.y = pose.position.y;

        // fix floor discards the boost translation, pulling the user back to where they
        // would be without it. Boost is horizontal-only, so this never fights the Y
        // correction computed above.
        let boost = self.boost.take_offset();
        pose.position.x -= boost.x;
        pose.position.z -= boost.z;
        self.boost_base = None;

        let _ = monado
            .ipc
            .set_reference_space_offset(ReferenceSpaceType::Stage, pose)
            .inspect_err(|e| log::warn!("Could not fix floor due to libmonado error: {e:?}"));

        let after =
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, anchor, &before, &after);
        }
    }

    /// Like `reset_offset`, but also throws away the translation contributed by space
    /// boost instead of carrying it through.
    pub fn full_reset(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        log::info!("Full playspace reset");
        self.boost.take_offset();
        self.boost_base = None;
        self.reset_offset(app, overlays);
    }

    /// Drops the boost translation and pulls the stage back by it, so the user ends up
    /// where they would have been had they never boosted.
    fn clear_boost(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        let boost = self.boost.take_offset();
        self.boost_base = None;

        if boost == Vec3A::ZERO {
            return;
        }

        let Some(monado) = &mut app.monado_state else {
            return;
        };

        let Ok(pose) = monado
            .ipc
            .get_reference_space_offset(ReferenceSpaceType::Stage)
            .log_err("Could not get reference space offset.")
        else {
            return;
        };

        let before =
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into());

        let mut after = before;
        after.translation -= boost;

        apply_offset(after, &mut monado.ipc);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &before, &after);
        }
    }

    /// One step of stick locomotion. Only runs when nothing else is moving the stage.
    fn update_boost(
        &mut self,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenXrOverlayData>,
    ) {
        if !self.boost.enabled() {
            return;
        }

        // whichever hand is pushing hardest wins, so the binding works on either side
        let stick = app
            .input_state
            .pointers
            .iter()
            .map(|p| Vec2::new(p.now.space_move_x, p.now.space_move_y))
            .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
            .unwrap_or(Vec2::ZERO);

        let Some(monado) = &mut app.monado_state else {
            return;
        };

        let base = if let Some(base) = self.boost_base {
            base
        } else {
            let Ok(pose) = monado
                .ipc
                .get_reference_space_offset(ReferenceSpaceType::Stage)
                .log_err("Could not get reference space offset.")
            else {
                return;
            };
            Affine3A::from_rotation_translation(pose.orientation.into(), pose.position.into())
        };

        let Some(delta) = self.boost.update(&SpaceBoostUpdateParams {
            dt: app.delta_time,
            config: &app.session.config,
            hmd: base * app.input_state.hmd,
            stick,
        }) else {
            // cache the base anyway: it is still correct, just unused this frame
            self.boost_base = Some(base);
            return;
        };

        let mut after = base;
        after.translation += delta;

        apply_offset(after, &mut monado.ipc);
        self.boost_base = Some(after);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &base, &after);
        }
    }

    pub fn save_center(&mut self, monado: &mut Monado) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by save center");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by save center");
            self.rotate = None;
        }

        let Ok(pose) = monado
            .get_reference_space_offset(ReferenceSpaceType::Stage)
            .inspect_err(|e| log::warn!("Could not fix floor due to libmonado error: {e:?}"))
        else {
            return;
        };

        let cur_rot: Quat = pose.orientation.into();
        let cur_pos: Vec3 = pose.position.into();

        let stage_offset = Affine3A::from_rotation_translation(cur_rot, cur_pos);
        self.playspace_state.openxr_space_center = stage_offset;
        let _ =
            save_playspace_state(&self.playspace_state).log_err("Could not save playspace state");
    }
}

fn apply_offset(transform: Affine3A, monado: &mut Monado) {
    let pose = Pose {
        position: transform.translation.into(),
        orientation: Quat::from_affine3(&transform).into(),
    };
    let _ = monado.set_reference_space_offset(ReferenceSpaceType::Stage, pose);
}
