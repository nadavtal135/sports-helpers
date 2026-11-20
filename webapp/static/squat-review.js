"use strict";

function openSquatReview(r, file, jobId) {
  const fps = r.meta.analysis_video.fps, offset = r.meta.trim.source_frame_offset;
  const footage = openTechniqueFootage(r, file, "A fully visible rep is needed for a movement observation.");
  const {view, showNotes, openNote} = footage;
  const form = $("review-targets");
  const savedKey = `squat-targets/${jobId}`;
  let review = r.review, controller, closed = false;
  const repName = (a) => a.rep ? `Rep ${a.rep}` : `Attempt ${a.attempt}`;
  const depth = (value) => ({above: "Hip above knee", "roughly level": "Hip roughly level with knee",
                              below: "Hip below knee", unknown: "Unknown"})[value];
  const cue = (a, phase, frame, measurement = "") => ({rep: a.rep, attempt: a.attempt, phase, source_frame: frame, measurement});
  const labels = {descent: "Descent", bottom: "Bottom", ascent: "Ascent", start: "Start", end: "End", peak_flexion: "Peak flexion"};
  const rowById = new Map(r.reps.map(a => [a.attempt, a]));

  function phaseButton(a, phase, value) {
    const frame = a[`source_frame_${phase}`];
    return frame == null ? `<span class="missing-phase">${labels[phase]} —</span>` :
      `<button type="button" data-phase="${phase}" data-attempt="${a.attempt}">
        ${labels[phase]} <small>${value || fmt(frame / fps, 2, " s")}</small></button>`;
  }
  function measurement(a, phase) {
    if (phase === "descent" || phase === "ascent") {
      const seconds = a[phase + "_s"];
      return seconds == null ? "Partial phase" : `${labels[phase]} time ${fmt(seconds, 2, " s")}`;
    }
    if (phase === "bottom") return `2D depth · ${depth(a.depth_2d)} · Torso lean ${fmt(a.torso_lean_at_bottom_deg, 1, "°")}`;
    return phase === "peak_flexion" ? `Projected knee flexion ${fmt(a.peak_knee_flexion_projected_deg, 1, "°")}` : "";
  }
  function showTimeline() {
    $("timeline").innerHTML = r.reps.map(a => {
      const issue = review.quality[a.attempt], checks = review.checks[a.attempt] || [];
      const goals = checks.map(c => {
        const label = c.metric[0].toUpperCase() + c.metric.slice(1);
        const actual = c.status === "unknown" ? c.difference : c.actual;
        const difference = c.metric !== "depth" && c.status !== "unknown" ? ` · ${c.difference}` : "";
        const text = `${label} target · ${c.status} · ${actual} (goal ${c.target})${difference}`;
        return c.source_frame != null ? `<button class="check" data-attempt="${a.attempt}" data-phase="${c.phase}">${esc(text)}</button>`
                                     : `<p class="muted">${esc(text)}</p>`;
      }).join("");
      return `<article class="rep-event"><div class="rep-head"><button class="rep-play" data-play="${a.attempt}"
        aria-label="Play ${repName(a)}">▶ ${repName(a)}</button><span class="muted">${esc(a.status)}</span></div>
        <div class="phase-strip" aria-label="${repName(a)} phases">
          ${["descent","bottom","ascent"].map(p => phaseButton(a, p)).join("")}</div>
        ${issue ? `<p class="quality-note">${esc(issue.detail)} <button class="text-button" data-issue="${a.attempt}">See frame</button></p>` : ""}
        ${goals}<details class="rep-measures"><summary>Rep measurements</summary>
          <p>2D depth · ${esc(depth(a.depth_2d))}</p><p>Torso lean at bottom · ${fmt(a.torso_lean_at_bottom_deg, 1, "°")}</p>
          <p>Peak projected knee flexion · ${fmt(a.peak_knee_flexion_projected_deg, 1, "°")}</p>
          <p>Descent ${fmt(a.descent_s, 2, " s")} · Ascent ${fmt(a.ascent_s, 2, " s")}</p>
          <div class="landmarks">${["start","bottom","end","peak_flexion"].map(p => phaseButton(a, p)).join("")}</div>
        </details></article>`;
    }).join("") || '<p class="rep-event muted">No visible squat attempt was detected.</p>';
  }

  $("timeline").onclick = event => {
    const b = event.target.closest("button");
    if (!b) return;
    $("observations").querySelectorAll("button").forEach(n => n.setAttribute("aria-pressed", "false"));
    if (b.dataset.issue) return view(review.quality[b.dataset.issue].evidence);
    const a = rowById.get(Number(b.dataset.attempt || b.dataset.play));
    if (b.dataset.play) {
      const start = cue(a, "review start", Math.round(a.review_start_s * fps));
      return view([start], 0, true, Math.round(a.review_end_s * fps));
    }
    const phase = b.dataset.phase;
    view([cue(a, phase, a[`source_frame_${phase}`], measurement(a, phase))]);
  };
  function displayReview(next) {
    review = next;
    showNotes(next);
    showTimeline();
    const targets = Object.entries(review.targets);
    show("target-report", targets.length > 0);
    const params = new URLSearchParams({...review.targets, tolerance: review.tempo_tolerance_s});
    $("target-report").href = `/api/jobs/${jobId}/squat-review?${params}`;
    $("target-summary").textContent = targets.length ? ` · ${targets.map(([k,v]) => k === "depth" ? depth(v) : `${k} ${v} s`).join(" · ")}` : " · Optional";
    if (review.strength || review.closer_look) openNote(review.strength ? "strength" : "closer_look", false);
    else view([{phase: "clip start", source_frame: offset}], 0, false);
  }
  function fillTargets(values) {
    for (const key of ["depth", "descent", "ascent"]) form.elements[key].value = values[key] ?? "";
    form.elements.tolerance.value = values.tolerance ?? 0.2;
  }
  async function applyTargets(values) {
    controller?.abort();
    const request = new AbortController();
    controller = request;
    const query = new URLSearchParams(Object.entries(values).filter(([,v]) => v !== ""));
    try {
      const next = await api(`/api/jobs/${jobId}/squat-review?${query}`, {signal: request.signal});
      if (closed || request !== controller) return;
      displayReview(next);
      fillTargets({...next.targets, tolerance: next.tempo_tolerance_s});
      try {
        if (Object.keys(next.targets).length) localStorage.setItem(savedKey, JSON.stringify(values));
        else localStorage.removeItem(savedKey);
      } catch { /* review still works when browser storage is unavailable */ }
      $("target-feedback").textContent = Object.keys(next.targets).length ? "Target comparisons are shown with each rep." : "Targets cleared.";
    } catch (err) {
      if (!closed && request === controller) $("target-feedback").textContent = `Could not apply targets. ${err.message}`;
    }
  }
  $("targets-panel").open = false;
  $("target-feedback").textContent = "";
  fillTargets({});
  form.onsubmit = e => { e.preventDefault(); applyTargets(Object.fromEntries(new FormData(form))); };
  $("clear-targets").onclick = () => { fillTargets({}); applyTargets({}); };
  $("timeline-title").textContent = "Rep timeline";
  const s = r.summary;
  $("set-summary").textContent = `${s.completed_reps} completed · ${s.incomplete_attempts} incomplete · ${s.uncertain_attempts} uncertain` +
    (s.selected_side ? ` · ${s.selected_side} side` : "");
  $("warnings").replaceChildren();
  $("charts").replaceChildren();
  $("limits").innerHTML = "Depth compares the visible hip and knee keypoints in 2D. Torso lean and knee flexion are image-plane angles. " +
    "Consistency describes how closely the measured reps repeat. Your chosen targets describe your intention. " +
    "Times refer to the original upload. Descent and ascent are timed to and from the bottom landmark, including time in a hold." +
    (r.warnings.length ? `<details><summary>Capture and tracking notes</summary>${r.warnings.map(w => `<p>${esc(w)}</p>`).join("")}</details>` : "");
  displayReview(review);
  try {
    const saved = JSON.parse(localStorage.getItem(savedKey));
    if (saved) applyTargets(saved);
  } catch { /* optional per-job browser preference */ }
  return () => {
    closed = true;
    controller?.abort();
    footage.close();
    $("timeline").onclick = form.onsubmit = $("clear-targets").onclick = null;
  };
}
