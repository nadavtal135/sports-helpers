"use strict";

function openJumpReview(r, file, jobId) {
  const footage = openTechniqueFootage(r, file, "There is too little visible movement for a technique observation.", true);
  const fps = r.meta.analysis_video.fps, offset = r.meta.trim.source_frame_offset;
  const timeline = $("timeline"), form = $("jump-target-form"), key = `jump-target/${jobId}`;
  const byId = new Map(r.attempts.map(a => [a.attempt, a]));
  const phases = {countermovement: "Countermovement", takeoff: "Takeoff", flight: "Flight",
    contact: "First contact", landing_bottom: "Deepest landing bend", recovery: "Settled recovery"};
  const targetNames = {descent_s: "Descent time", landing_bend_deg: "Added knee bend", landing_torso_deg: "Landing torso lean"};
  let review, controller, closed = false;

  function phaseButton(a, phase, label) {
    const cue = a.cues[phase];
    if (!cue) return `<span class="missing-phase">${label}<small>Unavailable</small></span>`;
    const limited = a.issues.some(q => q.phase === phase);
    return `<button type="button" data-phase="${phase}">${label}<small>${fmt(cue.source_frame / fps, 2, " s")}${limited ? " · uncertain" : ""}</small></button>`;
  }

  function attemptRow(a) {
    const check = review.checks[a.attempt];
    const issues = a.issues.map((q, i) => `<p class="quality-note">${esc(q.detail)}
      <button class="text-button" data-issue="${i}">Inspect frames</button></p>`).join("");
    const status = check ? (check.status === "met" ? "within range" : check.status === "unknown" ? "unknown" : check.status + " range") : "";
    const checkText = check ? `${targetNames[review.target.metric]} target · ${status} · ${check.measurement}` : "";
    const target = check && check.evidence.length ? `<button class="check" data-target>${esc(checkText)}</button>` :
      check ? `<p>${esc(checkText)}</p>` : "";
    const flight = a.metrics.flight_time_estimate_s;
    return `<article class="rep-event" data-attempt="${a.attempt}">
      <div class="rep-head"><button class="rep-play" data-play aria-label="Play attempt ${a.attempt}">▶ Attempt ${a.attempt}</button>
      <span class="muted">${a.status === "confirmed" ? "Observed takeoff and contact" : "Partial view"}</span></div>
      <div class="phase-strip" aria-label="Attempt ${a.attempt} phases">${Object.entries(phases).map(([p, label]) => phaseButton(a, p, label)).join("")}</div>
      ${a.frames.contact != null || a.step === "observed" ? `<p>${a.step === "observed" ? "A visible foot lifted and replanted after landing. " : ""}<button class="text-button" data-recovery>Replay recovery</button></p>` : ""}
      ${issues}${target}
      <details class="rep-measures"><summary>Attempt measurements</summary>
        ${a.cues.dip_start ? `<p>${phaseButton(a, "dip_start", "Dip start")}</p>` : ""}
        ${["countermovement", "takeoff", "contact", "landing_bottom"].filter(p => a.cues[p]?.measurement).map(p =>
          `<p><button class="text-button" data-phase="${p}">${esc(a.cues[p].phase)}</button> · ${esc(a.cues[p].measurement)}</p>`).join("")}
        <p>Estimated flight time · ${flight == null ? "Uncertain" : fmt(flight, 2, " s")}</p>
        <p>Recovery step · ${esc(a.step)}</p>
        ${a.cues.contact && a.cues.landing_bottom ? `<button class="text-button" data-bend>Compare contact and deepest bend</button>` : ""}
      </details></article>`;
  }

  function display(next) {
    review = next;
    footage.showNotes(next);
    timeline.innerHTML = r.attempts.map(attemptRow).join("") || '<p class="rep-event muted">No jump attempt available to review.</p>';
    const t = next.target;
    const unit = t?.metric === "descent_s" ? " s" : "°";
    $("jump-target-summary").textContent = t ? ` · ${targetNames[t.metric]} ${t.low}–${t.high}${unit}` : " · Optional";
    if (next.strength || next.closer_look) footage.openNote(next.strength ? "strength" : "closer_look", false);
    else footage.view([{source_frame: offset, phase: "Clip start", measurement: ""}], 0, false);
  }

  timeline.onclick = e => {
    const b = e.target.closest("button");
    if (!b) return;
    const a = byId.get(Number(b.closest("[data-attempt]").dataset.attempt));
    $("observations").querySelectorAll("button").forEach(n => n.setAttribute("aria-pressed", "false"));
    if (b.hasAttribute("data-phase")) return footage.view([a.cues[b.dataset.phase]]);
    if (b.hasAttribute("data-target")) return footage.view(review.checks[a.attempt].evidence);
    if (b.hasAttribute("data-bend")) return footage.view([a.cues.contact, a.cues.landing_bottom]);
    if (b.hasAttribute("data-issue")) {
      const q = a.issues[Number(b.dataset.issue)];
      return footage.view([{attempt: a.attempt, phase: "Visibility check", source_frame: q.source_frame, measurement: q.detail}]);
    }
    const recovery = b.hasAttribute("data-recovery");
    const first = recovery ? (a.frames.contact ?? a.frames.step_lift) : a.first_frame;
    footage.view([{attempt: a.attempt, phase: recovery ? "Recovery replay" : "Attempt start", source_frame: first}],
                 0, true, recovery ? (a.frames.recovery ?? a.last_frame) : a.last_frame);
  };

  async function apply(target) {
    controller?.abort();
    const request = controller = new AbortController();
    try {
      const q = new URLSearchParams(target || {});
      const result = await api(`/api/jobs/${jobId}/jump-review?${q}`, {signal: request.signal});
      if (closed || request !== controller) return;
      display(result);
      form.reset();
      if (result.target) {
        for (const [k, v] of Object.entries(result.target)) form.elements[k].value = v;
      }
      try {
        if (result.target) localStorage.setItem(key, JSON.stringify(result.target));
        else localStorage.removeItem(key);
      } catch { /* Browser storage is optional. */ }
      $("jump-target-message").textContent = result.target ? "Each attempt now shows how it compared." : "Target cleared.";
    } catch (error) {
      if (!closed && controller === request) $("jump-target-message").textContent = error.message;
    }
  }
  $("jump-targets").open = false;
  $("jump-target-message").textContent = "";
  form.reset();
  form.onsubmit = e => { e.preventDefault(); apply(Object.fromEntries(new FormData(form))); };
  $("clear-jump-target").onclick = () => apply(null);
  $("timeline-title").textContent = "Jump timeline";
  $("set-summary").textContent = `${r.summary.confirmed_jumps} confirmed · ${r.summary.partial_attempts} partial` +
    (r.summary.side ? ` · ${r.summary.side} side measured` : "");
  $("charts").replaceChildren();
  $("limits").textContent = "Takeoff and first contact are estimates from the shoe edge against the exposed floor when visible, or from ankle movement. " +
    "Use the shoes and floor to check the exact frame. Knee flexion and torso lean are side-view projections. " +
    "Times and frame numbers refer to the original clip on the analysis frame grid.";
  display(r.review);
  try {
    const saved = JSON.parse(localStorage.getItem(key));
    if (saved) apply(saved);
  } catch { /* Browser storage is optional. */ }
  return () => {
    closed = true;
    controller?.abort();
    footage.close();
    timeline.onclick = form.onsubmit = $("clear-jump-target").onclick = null;
  };
}
