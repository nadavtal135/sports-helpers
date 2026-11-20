"""Squat pose and phase overlay."""

from running.src import side_view_video


def render(info, analysis, dst, *, export_height, progress, cancelled):
    ends = {a.end for a in analysis.attempts if a.status == "completed"}
    side_view_video.render(info, analysis.points, analysis.phases, analysis.side, ends, dst,
                           export_height=export_height, progress=progress, cancelled=cancelled)
