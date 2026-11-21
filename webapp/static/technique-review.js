"use strict";
let activeTechniqueReview = null;
function leaveTechniqueReview() {
  if (activeTechniqueReview) activeTechniqueReview();
  activeTechniqueReview = null;
}

// Both renditions use the same analysis-frame clock. A stored pre-1.1 job can
// instead use its upload preview, whose clock starts at the original clip.
function techniquePlayer(video, caption, sources, info, offset, showFrames) {
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
    const stamp = fmt(frame / info.fps, 2, " s") + (showFrames ? ` · frame ${frame}` : "");
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
    step(by) {
      if (!source) return;
      stopFrame = null;
      const frame = Math.max(0, Math.min(info.n_frames - 1, Math.floor(position() * info.fps) + by));
      present(source, (frame + 0.5) / info.fps, false);
    },
    clear,
    close() {
      video.onloadedmetadata = video.onseeked = video.ontimeupdate = null;
      clear();
    },
  };
}

function openTechniqueFootage(r, file, emptyStrength, stepFrames = false) {
  const info = r.meta.analysis_video, fps = info.fps, offset = r.meta.trim.source_frame_offset;
  const clean = file(`${r.sport}_clean.mp4`);
  const sources = {
    pose: {url: file(`${r.sport}_annotated.mp4`).url + "?inline=1", start: 0},
    clean: clean ? {url: clean.url + "?inline=1", start: 0}
                 : {url: r.original_preview_url, start: offset / fps, preview: true},
  };
  const player = techniquePlayer($("result-video"), $("frame-caption"), sources, info, offset, stepFrames);
  const reference = techniquePlayer($("reference-video"), $("reference-caption"), sources, info, offset, stepFrames);
  const overlay = $("pose-overlay");
  overlay.checked = true;
  overlay.disabled = !sources.clean.url;
  let review;
  const repName = e => e.rep ? `Rep ${e.rep}` : `Attempt ${e.attempt}`;

  function view(evidence, index = 0, scroll = true, endFrame = null) {
    const current = evidence[index];
    player.select(current, overlay.checked, endFrame);
    const other = evidence[index === 0 ? 1 : 0];
    show("reference-frame", !!other);
    $("footage-pair").classList.toggle("compare", !!other);
    if (other) reference.select(other, overlay.checked); else reference.clear();
    const picker = $("evidence-choices");
    const choose = evidence.length > (stepFrames ? 1 : 2);
    show("evidence-choices", choose);
    if (choose) {
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

  function showNotes(next) {
    review = next;
    const cards = [["strength", "What did I do well?"], ["closer_look", "What should I look at more closely?"]];
    $("observations").innerHTML = cards.map(([key, question]) => {
      const note = review[key];
      if (!note) {
        const empty = key === "strength" ? emptyStrength
          : "There is no further difference to highlight in this set.";
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

  overlay.onchange = () => {
    player.toggle(overlay.checked);
    if (!$("reference-frame").hidden) reference.toggle(overlay.checked);
  };
  show("frame-steps", stepFrames);
  if (stepFrames) {
    $("previous-frame").onclick = () => player.step(-1);
    $("next-frame").onclick = () => player.step(1);
  }
  return {view, showNotes, openNote, close() {
    player.close(); reference.close();
    overlay.onchange = $("previous-frame").onclick = $("next-frame").onclick = null;
    show("reference-frame", false); show("evidence-choices", false); show("frame-steps", false);
    $("footage-pair").classList.remove("compare");
  }};
}
