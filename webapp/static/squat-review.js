"use strict";
let activeSquatReview = null;
function leaveSquatReview() {
  if (activeSquatReview) activeSquatReview();
  activeSquatReview = null;
}

// Both renditions use the same analysis-frame clock. A stored pre-1.1 job can
// instead use its upload preview, whose clock starts at the original clip.
function squatPlayer(video, caption, sources, info, offset) {
  let source, pending, cue, stopFrame;
  const duration = info.n_frames / info.fps;
  function position() {
    return pending ? pending.time : Math.max(0, video.currentTime - (source?.start || 0));
  }
  function updateCaption() {
    if (!source || pending) return;
    if (source.preview && (video.currentTime < source.start || video.currentTime >= source.start + duration)) {
      video.pause();
      video.currentTime = source.start + Math.min(duration - 0.5 / info.fps, Math.max(0, position()));
      return;
    }
    const frame = offset + Math.min(info.n_frames - 1, Math.max(0, Math.floor(position() * info.fps)));
    if (stopFrame != null && !video.seeking && frame >= stopFrame) {
      const end = stopFrame;
      stopFrame = null;
      cue = {...cue, phase: "review end", source_frame: end, measurement: ""};
      video.pause();
      video.currentTime = source.start + (end - offset + 0.5) / info.fps;
      if (frame !== end) return;
    }
    const stamp = fmt(frame / info.fps, 2, " s");
    if (cue && cue.source_frame === frame) {
      const who = cue.rep ? `Rep ${cue.rep}` : cue.attempt ? `Attempt ${cue.attempt}` : "Set";
      caption.textContent = [who, cue.phase.replaceAll("_", " "), stamp, cue.measurement].filter(Boolean).join(" · ");
    } else caption.textContent = `Original clip · ${stamp}`;
  }
  function apply() {
    if (!pending || video.readyState < 1) return;
    const {time, play} = pending;
    pending = null;
    video.currentTime = source.start + Math.min(duration - 0.5 / info.fps, Math.max(0, time));
    if (play) video.play().catch(() => {}); else video.pause();
    updateCaption();
  }
  function present(next, time, play) {
    pending = {time, play};
    if (source !== next) {
      video.pause();
      source = next;
      video.src = source.url;
      video.load();
    } else apply();
  }
  video.onloadedmetadata = apply;
  video.onseeked = video.ontimeupdate = updateCaption;
  function clear() {
    if (!source) return;
    pending = cue = source = stopFrame = null;
    video.pause();
    video.removeAttribute("src");
    video.load();
    caption.textContent = "";
  }
  return {
    select(evidence, overlay, endFrame = null) {
      cue = evidence;
      stopFrame = endFrame;
      present(overlay ? sources.pose : sources.clean, (cue.source_frame - offset + 0.5) / info.fps, endFrame != null);
    },
    toggle(overlay) {
      if (source) present(overlay ? sources.pose : sources.clean, position(), pending ? pending.play : !video.paused);
    },
    clear,
    close() {
      video.onloadedmetadata = video.onseeked = video.ontimeupdate = null;
      clear();
    },
  };
}

function openSquatReview(r, file, jobId) {
  const info = r.meta.analysis_video, fps = info.fps, offset = r.meta.trim.source_frame_offset;
  const clean = file("squat_clean.mp4");
  const sources = {
    pose: {url: file("squat_annotated.mp4").url + "?inline=1", start: 0},
    clean: clean ? {url: clean.url + "?inline=1", start: 0}
                 : {url: r.original_preview_url, start: offset / fps, preview: true},
  };
  const player = squatPlayer($("result-video"), $("frame-caption"), sources, info, offset);
  const reference = squatPlayer($("reference-video"), $("reference-caption"), sources, info, offset);
  const overlay = $("pose-overlay"), form = $("review-targets");
  const savedKey = `squat-targets/${jobId}`;
  let review = r.review, controller, closed = false;
  overlay.checked = true;
  overlay.disabled = !sources.clean.url;
  const repName = (a) => a.rep ? `Rep ${a.rep}` : `Attempt ${a.attempt}`;
  const depth = (value) => ({above: "Hip above knee", "roughly level": "Hip roughly level with knee",
                              below: "Hip below knee", unknown: "Unknown"})[value];
  const cue = (a, phase, frame, measurement = "") => ({rep: a.rep, attempt: a.attempt, phase, source_frame: frame, measurement});
  const labels = {descent: "Descent", bottom: "Bottom", ascent: "Ascent", start: "Start", end: "End", peak_flexion: "Peak flexion"};
  const rowById = new Map(r.reps.map(a => [a.attempt, a]));

  function view(evidence, index = 0, scroll = true, endFrame = null) {
    const current = evidence[index];
    player.select(current, overlay.checked, endFrame);
    const other = evidence[index === 0 ? 1 : 0];
    show("reference-frame", !!other);
    $("footage-pair").classList.toggle("compare", !!other);
    if (other) reference.select(other, overlay.checked); else reference.clear();
    const picker = $("evidence-choices");
    show("evidence-choices", evidence.length > 2);
    if (evidence.length > 2) {
      picker.innerHTML = `<label>Inspect a frame <select>${evidence.map((e, i) =>
        `<option value="${i}">${esc(repName(e))} · ${esc(e.phase)} · ${fmt(e.source_frame / fps, 2, " s")}</option>`
      ).join("")}</select></label>`;
      const select = picker.querySelector("select");
      select.value = String(index);
      select.onchange = () => view(evidence, Number(select.value), false);
    }
    const bounds = $("footage-pair").getBoundingClientRect();
    if (scroll && (bounds.top < 0 || bounds.bottom > innerHeight)) {
      $("footage-heading").scrollIntoView({block: "start"});
    }
  }

  function openNote(key, scroll = true) {
    const note = review[key];
    if (!note) return;
    $("observations").querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.note === key)));
    view(note.evidence, 0, scroll);
  }

  function showNotes() {
    const cards = [["strength", "What did I do well?"], ["closer_look", "What should I look at more closely?"]];
    $("observations").innerHTML = cards.map(([key, question]) => {
      const note = review[key];
      if (!note) {
        const empty = key === "strength" ? "A fully visible rep is needed for a movement observation."
          : "There is no further difference to highlight in the measured reps.";
        return `<section><h3>${question}</h3><p class="muted">${empty}</p></section>`;
      }
      const frames = note.evidence, phase = frames[0].phase.replaceAll("_", " ");
      const action = frames.length > 1 ? (frames.every(e => e.phase === frames[0].phase)
        ? `Compare ${phase} frames` : "Compare phase frames") : `View ${phase} frame`;
      return `<section><h3>${question}</h3><button class="observation" data-note="${key}" aria-pressed="false">
        <span class="observation-kind">${esc(note.kind)}</span><strong>${esc(note.title)}</strong>
        <span>${esc(note.detail)}</span><span class="evidence-action">${esc(action)} →</span></button></section>`;
    }).join("");
    $("observations").querySelectorAll("[data-note]").forEach(b => b.onclick = () => openNote(b.dataset.note));
  }

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
  overlay.onchange = () => {
    player.toggle(overlay.checked);
    if (!$("reference-frame").hidden) reference.toggle(overlay.checked);
  };
  function displayReview(next) {
    review = next;
    showNotes();
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
    player.close(); reference.close();
    $("timeline").onclick = overlay.onchange = form.onsubmit = $("clear-targets").onclick = null;
    show("reference-frame", false); show("evidence-choices", false);
    $("footage-pair").classList.remove("compare");
  };
}
