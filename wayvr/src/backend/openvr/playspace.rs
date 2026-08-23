use glam::{Affine3A, Quat, Vec2, Vec3, Vec3A, vec3a};
use ovr_overlay::{
    chaperone_setup::ChaperoneSetupManager,
    compositor::CompositorManager,
    sys::{EChaperoneConfigFile, ETrackingUniverseOrigin, HmdMatrix34_t},
};
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

use super::{helpers::Affine3AConvert, overlay::OpenVrOverlayData};

struct MoverData<T> {
    pose: Affine3A,
    hand: usize,
    hand_pose: T,
}

/// Space rotate is recomputed absolutely from the grab pose each frame, so it needs
/// more state than the incremental `MoverData`.
struct RotateData {
    /// chaperone working copy at the moment the gesture started
    initial_pose: Affine3A,
    /// working copy currently committed
    pending_pose: Affine3A,
    /// working copy committed on the previous frame
    previous_pose: Affine3A,
    hand: usize,
    /// hand orientation at the moment the gesture started
    initial_hand_pose: Quat,
    /// point the rotation pivots around, fixed for the whole gesture
    initial_pivot: Vec3A,
}

pub(super) struct PlayspaceMover {
    universe: ETrackingUniverseOrigin,
    drag: Option<MoverData<Vec3A>>,
    rotate: Option<RotateData>,
    gravity: SpaceGravity,
    boost: SpaceBoost,
    /// cached working copy for the boost path. None means it has gone stale
    boost_base: Option<Affine3A>,
    playspace_state: PlayspaceState,
}

impl PlayspaceMover {
    pub fn new() -> Self {
        Self {
            universe: ETrackingUniverseOrigin::TrackingUniverseRawAndUncalibrated,
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
        chaperone_mgr: &mut ChaperoneSetupManager,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
        task: PlayspaceTask,
    ) {
        match task {
            PlayspaceTask::FixFloor => {
                self.fix_floor(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::Reset => {
                self.reset_offset(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::FullReset => {
                self.full_reset(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::Recenter => {
                self.recenter(chaperone_mgr, app, overlays);
            }
            PlayspaceTask::SaveCenter => {
                self.save_center(chaperone_mgr);
            }
            PlayspaceTask::ResetCenter => {
                self.playspace_state.openvr_space_center = Affine3A::IDENTITY;
            }
        }
    }

    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
    pub fn update(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
        app: &mut AppState,
    ) {
        let universe = self.universe.clone();

        // `space_fling` toggles space gravity, `space_boost` toggles stick locomotion.
        // Both are session-only and are not written back to the config file.
        if app
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_fling && !p.before.space_fling)
        {
            let enabled = !app.session.config.space_gravity_enabled;
            app.session.config.space_gravity_enabled = enabled;
            if !enabled {
                self.gravity.reset();
            }
            log::info!("Space gravity {}", if enabled { "enabled" } else { "disabled" });
        }

        if app
            .input_state
            .pointers
            .iter()
            .any(|p| p.now.space_boost && !p.before.space_boost)
        {
            let enabled = self.boost.toggle();
            log::info!("Space boost {}", if enabled { "enabled" } else { "disabled" });
        }

        if let Some(data) = self.rotate.as_mut() {
            let pointer = &app.input_state.pointers[data.hand];
            if !pointer.now.space_rotate {
                self.rotate = None;
                log::info!("End space rotate");
                return;
            }

            // recomputed absolutely from the grab pose every frame, so the gesture cannot
            // accumulate drift over its lifetime
            let new_hand = Quat::from_affine3(&(data.pending_pose * pointer.raw_pose)).normalize();
            let mut delta = (new_hand * data.initial_hand_pose.conjugate()).normalize();
            if delta.w < 0.0 {
                // take the shortest path around the rotation
                delta = -delta;
            }

            // OpenVR can only ever rotate yaw
            let rel_y = f32::atan2(
                2.0 * delta.y.mul_add(delta.w, delta.x * delta.z),
                2.0f32.mul_add(delta.w.mul_add(delta.w, delta.x * delta.x), -1.0),
            );

            let mut space_transform = Affine3A::from_rotation_y(rel_y);

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

            if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
                // `space_transform` is absolute now, so the chaperone needs this frame's
                // delta rather than the whole rotation
                let frame_delta = data.previous_pose.inverse() * data.pending_pose;
                apply_chaperone_transform(frame_delta.inverse(), chaperone_mgr);
            }

            data.previous_pose = data.pending_pose;

            set_working_copy(&universe, chaperone_mgr, &data.pending_pose);
            chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        } else {
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_rotate {
                    let Some(mat) = get_working_copy(&universe, chaperone_mgr) else {
                        log::warn!("Can't space rotate - failed to get zero pose");
                        return;
                    };

                    let initial_hand_pose =
                        Quat::from_affine3(&(mat * pointer.raw_pose)).normalize();
                    let initial_pivot = mat.transform_point3a(app.input_state.hmd.translation);

                    self.rotate = Some(RotateData {
                        initial_pose: mat,
                        pending_pose: mat,
                        previous_pose: mat,
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

        if let Some(data) = self.drag.as_mut() {
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
                // hand this frame's movement to gravity so it can fling from it
                let drag_pose = data.pose;
                self.gravity
                    .mark_end_drag(&app.session.config, relative_pos, drag_pose, app.delta_time);
                self.drag = None;
                log::info!("End space drag");
                return;
            }

            if relative_pos.length_squared() > 1000.0 {
                log::warn!("Space drag too fast, ignoring");
                return;
            }

            let overlay_offset = data.pose.inverse().transform_vector3a(relative_pos) * -1.0;
            let before_pose = data.pose;
            data.pose.translation += relative_pos;
            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(overlays, &mut app.anchor, &before_pose, &data.pose);
            }
            data.hand_pose = new_hand;

            if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
                apply_chaperone_offset(overlay_offset, chaperone_mgr);
            }
            set_working_copy(&universe, chaperone_mgr, &data.pose);
            chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        } else {
            for (i, pointer) in app.input_state.pointers.iter().enumerate() {
                if pointer.now.space_drag {
                    let Some(mat) = get_working_copy(&universe, chaperone_mgr) else {
                        log::warn!("Can't space drag - failed to get zero pose");
                        return;
                    };
                    let hand_pos = mat.transform_point3a(pointer.raw_pose.translation);
                    self.drag = Some(MoverData {
                        pose: mat,
                        hand: i,
                        hand_pose: hand_pos,
                    });
                    self.rotate = None;
                    log::info!("Start space drag");
                    return;
                }
            }
        }

        for pointer in &app.input_state.pointers {
            if pointer.now.space_reset && !pointer.before.space_reset {
                self.reset_offset(chaperone_mgr, app, overlays);
                log::info!("Space reset");
                return;
            }
        }

        if let Some(res) = self.gravity.update(SpaceGravityUpdateParams {
            dt: app.delta_time,
            dragging: self.drag.is_some(),
            config: &app.session.config,
            floor_height: app.session.config.space_gravity_floor_height,
        }) {
            if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
                let moved = res.playspace_pose.translation - res.previous_pose.translation;
                let overlay_offset = res.previous_pose.inverse().transform_vector3a(moved) * -1.0;
                apply_chaperone_offset(overlay_offset, chaperone_mgr);
            }

            set_working_copy(&universe, chaperone_mgr, &res.playspace_pose);
            chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);

            if !app.session.config.space_drag_affects_world {
                playspace_common::shift_world(
                    overlays,
                    &mut app.anchor,
                    &res.previous_pose,
                    &res.playspace_pose,
                );
            }

            self.boost_base = None;
        }

        if self.gravity.just_landed() {
            // landing discards the boost translation and pulls the user back
            self.clear_boost(chaperone_mgr, app, overlays);
        }

        if self.drag.is_none() && self.rotate.is_none() && !self.gravity.is_active() {
            self.update_boost(chaperone_mgr, app, overlays);
        } else {
            self.boost_base = None;
        }
    }

    pub fn reset_offset(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        let Some(before) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't reset offset - failed to get zero pose");
            return;
        };

        let mut xform = self.playspace_state.openvr_space_center;
        if self.universe == ETrackingUniverseOrigin::TrackingUniverseSeated {
            xform.translation.y -= 1.7;
        }

        self.gravity.reset();
        // the boost translation deliberately survives a plain reset; only a full reset
        // (or fix floor / a gravity landing) throws it away
        xform.translation += self.boost.offset();
        self.boost_base = Some(xform);

        set_working_copy(&self.universe, chaperone_mgr, &xform);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &before, &xform);
        }

        if self.drag.is_some() {
            log::info!("Space drag interrupted by manual reset");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by manual reset");
            self.rotate = None;
        }
    }

    pub fn fix_floor(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        let input = &app.input_state;
        let anchor = &mut app.anchor;
        let y1 = input.pointers[0].pose.translation.y;
        let y2 = input.pointers[1].pose.translation.y;
        let Some(mut mat) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't fix floor - failed to get zero pose");
            return;
        };
        let offset = y1.min(y2) - 0.03;
        let before = mat;
        mat.translation.y += offset;
        self.playspace_state.openvr_space_center.translation.y = mat.translation.y;

        // fix floor discards the boost translation. Boost is horizontal-only, so this
        // never fights the Y correction computed above.
        let boost = self.boost.take_offset();
        mat.translation -= boost;
        self.boost_base = None;

        set_working_copy(&self.universe, chaperone_mgr, &mat);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, anchor, &before, &mat);
        }

        if self.drag.is_some() {
            log::info!("Space drag interrupted by fix floor");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by fix floor");
            self.rotate = None;
        }
    }

    pub fn recenter(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by recenter");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by recenter");
            self.rotate = None;
        }

        self.gravity.reset();
        // recenter keeps the boost translation but moves the stage, so the cache is stale
        self.boost_base = None;

        let input = &app.input_state;
        let anchor = &mut app.anchor;

        let Some(mat) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't recenter - failed to get zero pose");
            return;
        };

        let before = mat;

        let cur_rot: Quat = Quat::from_affine3(&mat);
        let cur_pos: Vec3 = Vec3::from(mat.translation);

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

        let new_mat = if horiz_len_sq > f32::EPSILON {
            Affine3A::from_rotation_translation(new_rot, new_pos)
        } else {
            let mut m = mat;
            m.translation = new_pos.into();
            m
        };

        set_working_copy(&self.universe, chaperone_mgr, &new_mat);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, anchor, &before, &new_mat);
        }
    }

    /// Like `reset_offset`, but also throws away the translation contributed by space
    /// boost instead of carrying it through.
    pub fn full_reset(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        log::info!("Full playspace reset");
        self.boost.take_offset();
        self.boost_base = None;
        self.reset_offset(chaperone_mgr, app, overlays);
    }

    /// Drops the boost translation and pulls the stage back by it, so the user ends up
    /// where they would have been had they never boosted.
    fn clear_boost(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
    ) {
        let boost = self.boost.take_offset();
        self.boost_base = None;

        if boost == Vec3A::ZERO {
            return;
        }

        let Some(before) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't clear space boost - failed to get zero pose");
            return;
        };

        let mut after = before;
        after.translation -= boost;

        if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
            let overlay_offset = before.inverse().transform_vector3a(-boost) * -1.0;
            apply_chaperone_offset(overlay_offset, chaperone_mgr);
        }

        set_working_copy(&self.universe, chaperone_mgr, &after);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &before, &after);
        }
    }

    /// One step of stick locomotion. Only runs when nothing else is moving the stage.
    fn update_boost(
        &mut self,
        chaperone_mgr: &mut ChaperoneSetupManager,
        app: &mut AppState,
        overlays: &mut OverlayWindowManager<OpenVrOverlayData>,
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

        let base = if let Some(base) = self.boost_base {
            base
        } else {
            let Some(base) = get_working_copy(&self.universe, chaperone_mgr) else {
                log::warn!("Can't space boost - failed to get zero pose");
                return;
            };
            base
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

        if self.universe == ETrackingUniverseOrigin::TrackingUniverseStanding {
            let overlay_offset = base.inverse().transform_vector3a(delta) * -1.0;
            apply_chaperone_offset(overlay_offset, chaperone_mgr);
        }

        set_working_copy(&self.universe, chaperone_mgr, &after);
        chaperone_mgr.commit_working_copy(EChaperoneConfigFile::EChaperoneConfigFile_Live);
        self.boost_base = Some(after);

        if !app.session.config.space_drag_affects_world {
            playspace_common::shift_world(overlays, &mut app.anchor, &base, &after);
        }
    }

    pub fn playspace_changed(
        &mut self,
        compositor_mgr: &mut CompositorManager,
        _chaperone_mgr: &mut ChaperoneSetupManager,
    ) {
        let new_universe = compositor_mgr.get_tracking_space();
        if new_universe != self.universe {
            log::info!(
                "Playspace changed: {} -> {}",
                universe_str(&self.universe),
                universe_str(&new_universe)
            );
            self.universe = new_universe;
        }

        if self.drag.is_some() {
            log::info!("Space drag interrupted by external change");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by external change");
            self.rotate = None;
        }
    }

    pub fn get_universe(&self) -> ETrackingUniverseOrigin {
        self.universe.clone()
    }

    pub fn save_center(&mut self, chaperone_mgr: &mut ChaperoneSetupManager) {
        if self.drag.is_some() {
            log::info!("Space drag interrupted by save center");
            self.drag = None;
        }
        if self.rotate.is_some() {
            log::info!("Space rotate interrupted by save center");
            self.rotate = None;
        }

        let Some(mat) = get_working_copy(&self.universe, chaperone_mgr) else {
            log::warn!("Can't save center - failed to get zero pose");
            return;
        };

        self.playspace_state.openvr_space_center = mat;
        let _ =
            save_playspace_state(&self.playspace_state).log_err("Could not save playspace state");
    }
}

const fn universe_str(universe: &ETrackingUniverseOrigin) -> &'static str {
    match universe {
        ETrackingUniverseOrigin::TrackingUniverseSeated => "Seated",
        ETrackingUniverseOrigin::TrackingUniverseStanding => "Standing",
        ETrackingUniverseOrigin::TrackingUniverseRawAndUncalibrated => "Raw",
    }
}

fn get_working_copy(
    universe: &ETrackingUniverseOrigin,
    chaperone_mgr: &mut ChaperoneSetupManager,
) -> Option<Affine3A> {
    chaperone_mgr.revert_working_copy();
    let mat = match universe {
        ETrackingUniverseOrigin::TrackingUniverseStanding => {
            chaperone_mgr.get_working_standing_zero_pose_to_raw_tracking_pose()
        }
        _ => chaperone_mgr.get_working_seated_zero_pose_to_raw_tracking_pose(),
    };
    mat.map(|m| m.to_affine())
}

fn set_working_copy(
    universe: &ETrackingUniverseOrigin,
    chaperone_mgr: &mut ChaperoneSetupManager,
    mat: &Affine3A,
) {
    let mat = HmdMatrix34_t::from_affine(mat);
    match universe {
        ETrackingUniverseOrigin::TrackingUniverseStanding => {
            chaperone_mgr.set_working_standing_zero_pose_to_raw_tracking_pose(&mat);
        }
        _ => chaperone_mgr.set_working_seated_zero_pose_to_raw_tracking_pose(&mat),
    }
}

fn apply_chaperone_offset(offset: Vec3A, chaperone_mgr: &mut ChaperoneSetupManager) {
    let mut quads = chaperone_mgr.get_live_collision_bounds_info();
    for quad in &mut quads {
        quad.vCorners.iter_mut().for_each(|corner| {
            corner.v[0] += offset.x;
            corner.v[2] += offset.z;
        });
    }
    chaperone_mgr.set_working_collision_bounds_info(quads.as_mut_slice());
}

fn apply_chaperone_transform(transform: Affine3A, chaperone_mgr: &mut ChaperoneSetupManager) {
    let mut quads = chaperone_mgr.get_live_collision_bounds_info();
    for quad in &mut quads {
        quad.vCorners.iter_mut().for_each(|corner| {
            let coord = transform.transform_point3a(Vec3A::from_slice(&corner.v));
            corner.v[0] = coord.x;
            corner.v[2] = coord.z;
        });
    }
    chaperone_mgr.set_working_collision_bounds_info(quads.as_mut_slice());
}
