"""Unchanged velocity-update block from the preserved VisualOdometry.process.

Only the wrapper function is new. Complete modules remain in origin/.
"""

def archived_velocity_update(self):
    if len(self.pose_history) > 0:
        prev_pos = self.pose_history[-1][:3, 3]
        self.state.velocity = self.state.position - prev_pos
