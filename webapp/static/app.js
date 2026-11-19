"use strict";
const $ = (id) => document.getElementById(id);
const state = { config: null, sport: null, upload: null, job: null, poll: null, lastRequest: null };

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1, unit = "") => (v === null || v === undefined || Number.isNaN(v)) ? "—" : `${Number(v).toFixed(d)}${unit}`;
const show = (id, on = true) => { $(id).hidden = !on; };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* not json */ }
  if (!res.ok) throw new Error((body && body.detail) || `${res.status} ${res.statusText}`);
  return body;
}

// ── boot ────────────────────────────────────────────────────────────────────
async function boot() {
  try {
    state.config = await api("/api/config");
  } catch (e) {
    $("mode-banner").textContent = `Could not reach the server: ${e.message}`;
    $("mode-banner").className = "banner error"; show("mode-banner");
    return;
  }
  const c = state.config;
  const badge = $("mode-badge");
  badge.textContent = c.mode === "live" ? "Live mode" : "Fixture mode";
  badge.className = `badge ${c.mode}`;
  if (c.mode === "fixture") {
    $("mode-banner").innerHTML = "<strong>Fixture mode.</strong> No API key is configured, so no model calls are made. " +
      "You can run the full flow on a <strong>synthetic sample clip</strong> with simulated model replies. " +
      "Your own uploads can be previewed but not analyzed until <code>VLMRUN_API_KEY</code> is set on the server.";
    show("mode-banner");
  }
  if (!c.ffmpeg.ok) {
    $("ffmpeg-banner").textContent = "FFmpeg/ffprobe were not found on the server, so uploads and analysis will fail. Install FFmpeg and restart.";
    show("ffmpeg-banner");
  }
  $("upload-limits").textContent = `Up to ${c.max_upload_mb} MB. MP4/MOV from a phone works.` +
    (c.mode === "live" ? ` Live analysis is limited to ${c.max_live_seconds}s per run (trim longer clips).` : "");
  $("provider-name").textContent = c.provider;
  $("capture-tip").textContent = c.sports.squat.guidance.join(" ");
  $("privacy-note").textContent = c.mode === "live"
    ? `Live analysis uploads the (trimmed) video to the model provider, ${c.provider}, for pose estimation and segmentation. The API key stays on the server. The synthetic sample is never sent.`
    : "In fixture mode nothing leaves this server.";
  $("sample-note").textContent = "Synthetic footage with known outcomes; analyzed with simulated model replies.";
  await restoreFromUrl();
}

// A reload (or a shared link on the same server) brings back the job in the URL.
async function restoreFromUrl() {
  const m = location.hash.match(/job=([0-9a-f]{32})/);
  if (!m) return;
  try { state.job = await api(`/api/jobs/${m[1]}`); }
  catch { history.replaceState(null, "", location.pathname); return; }
  state.sport = state.job.sport;
  document.querySelectorAll(".sport").forEach((b) => b.classList.toggle("selected", b.dataset.sport === state.sport));
  if (["queued", "running"].includes(state.job.status)) { watchJob(); return; }
  await pollJob();
}

// ── 1. sport ────────────────────────────────────────────────────────────────
document.querySelectorAll(".sport").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".sport").forEach((b) => b.classList.toggle("selected", b === btn));
  state.sport = btn.dataset.sport;
  show("step-upload");
  show("capture-tip", state.sport === "squat" && !state.upload);
  if (state.upload) renderSetup();
}));

// ── 2. upload ───────────────────────────────────────────────────────────────
function uploadStatus(text, kind = "") {
  const el = $("upload-status");
  el.textContent = text; el.className = `status ${kind}`; el.hidden = !text;
}

function postWithProgress(url, form, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) onProgress(e.loaded / e.total); };
    xhr.upload.onload = () => onProgress(1);
    xhr.onload = () => {
      let body = null;
      try { body = JSON.parse(xhr.responseText); } catch { /* not json */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error((body && body.detail) || `${xhr.status} ${xhr.statusText}`));
    };
    xhr.onerror = () => reject(new Error("network error: is the server still running?"));
    xhr.send(form);
  });
}

async function sendFile(file) {
  if (!file) return;
  const mb = (file.size / 2 ** 20).toFixed(1);
  if (file.size > state.config.max_upload_mb * 2 ** 20) {
    uploadStatus(`${file.name} is ${mb} MB; the limit is ${state.config.max_upload_mb} MB. Trim it on the phone first.`, "error");
    return;
  }
  const form = new FormData(); form.append("file", file);
  show("upload-bar");
  try {
    setUpload(await postWithProgress("/api/uploads", form, (f) => {
      $("upload-fill").style.width = `${Math.round(f * 100)}%`;
      uploadStatus(f < 1 ? `Uploading ${file.name}: ${Math.round(f * 100)}% of ${mb} MB`
                         : "Upload complete. Checking the video and preparing a preview…");
    }));
    uploadStatus("");
  } catch (e) { uploadStatus(`Upload failed: ${e.message}`, "error"); }
  show("upload-bar", false);
  $("upload-fill").style.width = "0";
}

$("file").addEventListener("change", (e) => sendFile(e.target.files[0]));
const dz = $("dropzone");
dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("over"); });
dz.addEventListener("dragleave", () => dz.classList.remove("over"));
dz.addEventListener("drop", (e) => { e.preventDefault(); dz.classList.remove("over"); sendFile(e.dataTransfer.files[0]); });
$("use-sample").addEventListener("click", async () => {
  if (!state.sport) return;
  uploadStatus("Preparing the synthetic sample clip (the first time takes a few seconds)…");
  try { setUpload(await api(`/api/samples/${state.sport}`, { method: "POST" })); uploadStatus(""); }
  catch (e) { uploadStatus(`Could not prepare the sample: ${e.message}`, "error"); }
});

function setUpload(u) { state.upload = u; renderSetup(); }

// ── 3. setup ────────────────────────────────────────────────────────────────
function renderSetup() {
  const u = state.upload, c = state.config, sport = state.sport;
  show("capture-tip", false);
  show("step-setup");
  $("preview").src = `/api/uploads/${u.id}/preview`;
  $("clip-info").textContent = `${u.sample_of ? "Synthetic sample · " : ""}${u.filename} · ${u.width}×${u.height} · ${fmt(u.fps, 2)} fps · ${fmt(u.duration, 1)} s`;
  $("trim-start").value = 0; $("trim-start").max = u.duration;
  $("trim-end").value = u.duration.toFixed(1); $("trim-end").max = u.duration;
  updateTrim();
  const spec = c.sports[sport];
  $("guidance").innerHTML = spec.guidance.map((g) => `<li>${esc(g)}</li>`).join("");
  show("settings-title", Object.keys(spec.fields).length > 0);
  $("settings").innerHTML = Object.entries(spec.fields).map(([key, f]) => {
    let input;
    if (f.type === "choice") {
      input = `<select name="${key}">${f.choices.map((ch, i) => `<option value="${esc(ch)}">${esc((f.labels || [])[i] ?? ch)}</option>`).join("")}</select>`;
    } else {
      input = `<input name="${key}" type="number" step="any" min="${f.min}" max="${f.max}" value="${f.default}">`;
    }
    return `<label>${esc(f.label)}${input}${f.help ? `<span class="help">${esc(f.help)}</span>` : ""}</label>`;
  }).join("");
  const wrongSport = u.sample_of && u.sample_of !== sport;
  const blocked = !u.sample_of && c.mode === "fixture";
  show("consent-row", !u.sample_of && c.mode === "live");
  const err = $("setup-error");
  if (wrongSport) { err.textContent = `This is the ${u.sample_of} sample. Choose ${u.sample_of} above, or load the ${sport} sample.`; }
  else if (blocked) { err.textContent = "Fixture mode can only analyze the synthetic sample clip: simulated replies would say nothing about your video. Set VLMRUN_API_KEY on the server to analyze your own footage, or use the sample."; }
  err.hidden = !(wrongSport || blocked);
  $("start").disabled = wrongSport || blocked;
}

function updateTrim() {
  const a = parseFloat($("trim-start").value) || 0, b = parseFloat($("trim-end").value) || state.upload.duration;
  const len = b - a;
  $("trim-len").textContent = len >= 2 ? `${len.toFixed(1)} s selected` : "select at least 2 s";
}
["trim-start", "trim-end"].forEach((id) => $(id).addEventListener("input", updateTrim));
$("set-start").addEventListener("click", () => { $("trim-start").value = $("preview").currentTime.toFixed(1); updateTrim(); });
$("set-end").addEventListener("click", () => { $("trim-end").value = $("preview").currentTime.toFixed(1); updateTrim(); });
$("change-clip").addEventListener("click", () => {
  state.upload = null; show("step-setup", false); $("file").value = "";
  show("capture-tip", state.sport === "squat");
});

$("start").addEventListener("click", () => {
  const settings = {};
  new FormData($("settings")).forEach((v, k) => { settings[k] = v; });
  const end = parseFloat($("trim-end").value);
  state.lastRequest = {
    upload_id: state.upload.id, sport: state.sport, settings,
    trim_start: parseFloat($("trim-start").value) || 0,
    trim_end: Number.isFinite(end) && end < state.upload.duration - 0.05 ? end : null,
    consent: $("consent").checked,
  };
  startJob();
});

// ── 4. progress ─────────────────────────────────────────────────────────────
async function startJob() {
  const err = $("setup-error");
  err.hidden = true;
  $("start").disabled = true;
  try {
    state.job = await api("/api/jobs", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(state.lastRequest) });
  } catch (e) {
    err.textContent = e.message; err.hidden = false; $("start").disabled = false;
    return;
  }
  history.replaceState(null, "", `#job=${state.job.id}`);
  watchJob();
}

function watchJob() {
  ["step-sport", "step-upload", "step-setup", "step-results", "step-ended"].forEach((id) => show(id, false));
  show("step-progress");
  $("cancel").disabled = false;
  renderProgress(state.job);
  state.poll = setInterval(pollJob, 800);
}

function renderProgress(job) {
  $("bar-fill").style.width = `${Math.round(job.progress * 100)}%`;
  $("progress-msg").textContent = `${Math.round(job.progress * 100)}% · ${job.message}`;
  let seen = false;
  $("stage-list").innerHTML = job.stages.map((s) => {
    const active = s.id === job.stage;
    const cls = active ? "active" : (!seen && job.stage ? "done" : "");
    if (active) seen = true;
    return `<li class="${cls}">${esc(s.label)}</li>`;
  }).join("");
}

async function pollJob() {
  let job;
  try { job = await api(`/api/jobs/${state.job.id}`); } catch (e) { $("progress-msg").textContent = `Lost contact with the server: ${e.message}`; return; }
  state.job = job;
  renderProgress(job);
  if (["completed", "failed", "cancelled"].includes(job.status)) {
    clearInterval(state.poll);
    show("step-progress", false);
    if (job.status === "completed") return showResults();
    $("ended-title").textContent = job.status === "cancelled" ? "Analysis cancelled" : "Analysis failed";
    $("ended-msg").textContent = job.status === "cancelled" ? job.message : (job.error || job.message);
    $("ended-msg").className = job.status === "failed" ? "status error" : "status";
    $("retry").hidden = !state.lastRequest;   // a job restored from the URL has no request to repeat
    show("step-ended");
  }
}

$("cancel").addEventListener("click", async () => {
  $("cancel").disabled = true;
  $("progress-msg").textContent = "Cancelling…";
  try { await api(`/api/jobs/${state.job.id}/cancel`, { method: "POST" }); } catch (e) { $("progress-msg").textContent = e.message; }
});
$("retry").addEventListener("click", () => { show("step-ended", false); startJob(); });
const restart = () => { history.replaceState(null, "", location.pathname); ["step-ended", "step-results", "step-progress"].forEach((id) => show(id, false)); show("step-sport"); show("step-upload"); if (state.upload) renderSetup(); };
$("restart").addEventListener("click", restart);
$("new-analysis").addEventListener("click", restart);

// ── 5. results ──────────────────────────────────────────────────────────────
async function showResults() {
  let r;
  try { r = await api(`/api/jobs/${state.job.id}/result`); }
  catch (e) { $("ended-title").textContent = "Could not load results"; $("ended-msg").textContent = e.message; show("step-ended"); return; }
  show("step-sport"); show("step-upload");
  show("step-results");
  show("sim-banner", !!r.meta.simulated);
  const file = (n) => r.files.find((f) => f.name === n);
  const trim0 = r.sport === "squat"
    ? r.meta.trim.source_frame_offset / r.meta.analysis_video.fps : r.meta.trim.start_s || 0;
  const vid = $("result-video");
  const v = file(`${r.sport}_annotated.mp4`);
  vid.src = v ? `${v.url}?inline=1` : "";
  const seek = (t, lead = 0.3, play = true) => {
    vid.currentTime = Math.max(0, t - trim0 - lead);
    if (play) vid.play().catch(() => {}); else vid.pause();
  };
  $("warnings").innerHTML = (r.warnings || []).map((w) => `<div class="warn-item">⚠ ${esc(w)}</div>`).join("");
  if (r.sport === "basketball") renderBasketball(r, file, seek);
  else if (r.sport === "squat") renderSquat(r, seek);
  else renderRunning(r, file, seek);
  $("downloads").innerHTML = r.files.map((f) => `<li><a href="${f.url}">${esc(f.name)}</a> <span class="muted small">${(f.bytes / 1024).toFixed(0)} KB</span></li>`).join("");
  if (r.files.length > 1) {
    $("downloads").insertAdjacentHTML("afterbegin", `<li><a href="/api/jobs/${state.job.id}/bundle.zip"><strong>Everything (zip)</strong></a></li>`);
  }
  $("step-results").scrollIntoView({ behavior: "smooth" });
}

const tile = (k, v, d = "", cls = "") => `<div class="tile ${cls}"><div class="k">${esc(k)}</div><div class="v">${v}</div><div class="d">${d}</div></div>`;

function renderBasketball(r, file, seek) {
  const s = r.stats;
  $("tiles").innerHTML = [
    tile("Shots found", s.attempts),
    tile("Makes", s.makes, "", "make"),
    tile("Misses", s.misses, "", "miss"),
    tile("Unknown", s.unknown, "not judged; not misses", "unknown"),
    tile("FG%", s.fg_pct === null ? "—" : `${Math.round(s.fg_pct * 100)}%`,
      `${s.makes} makes ÷ ${s.judged} judged (makes + misses)${s.unknown ? `; ${s.unknown} unknown excluded` : ""}`),
  ].join("");
  $("timeline-title").textContent = "Shot timeline";
  $("timeline").innerHTML = r.shots.length ? r.shots.map((sh) => {
    const extra = [];
    if (sh.distance_ft_estimate !== null) extra.push(`~${sh.distance_ft_estimate} ft (estimate)${sh.three_point_estimate ? ", likely a three" : ""}`);
    if (sh.side_on && sh.elbow_set_projected_deg !== null) extra.push(`projected elbow ${fmt(sh.elbow_set_projected_deg, 0)}° set → ${fmt(sh.elbow_release_projected_deg, 0)}° release`);
    if (sh.release_method === "pose") extra.push("release time from pose (ball not seen leaving the hand)");
    return `<div class="event" data-t="${sh.release_s}"><div class="t">${fmt(sh.release_s, 1)} s</div>
      <div><strong>Shot ${sh.shot}</strong> <span class="pill ${sh.result}">${sh.result.toUpperCase()}</span> ${esc(sh.reason)}
      <ul>${sh.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}${extra.map((e) => `<li>${esc(e)}</li>`).join("")}</ul></div></div>`;
  }).join("") : '<p class="muted" style="padding:10px">No shots were detected.</p>';
  $("timeline").querySelectorAll(".event").forEach((el) => el.addEventListener("click", () => seek(parseFloat(el.dataset.t))));
  const chart = file("shot_chart.png"), paths = file("ball_paths.png");
  $("charts").innerHTML =
    (chart ? `<figure><img src="${chart.url}?inline=1" alt="Shot chart"><figcaption>Shot chart. Locations are <strong>estimates</strong> from a single camera, scaled by the shooter height and field-of-view settings. Hollow amber = unknown.</figcaption></figure>` : "") +
    (paths ? `<figure><img src="${paths.url}?inline=1" alt="Ball paths at the hoop"><figcaption>Every tracked ball path against the rim line: what each call was read from.</figcaption></figure>` : "");
  $("limits").innerHTML = "<strong>Limits.</strong> One shooter, one ball, a still camera. Shots are found from the pose, so a shot out of frame is missed. " +
    "Unknown means the ball track could not support a call (for example the ball was lost after crossing the rim); it is excluded from FG%, and it neither extends nor breaks a streak. " +
    "Court distance is an estimate. Elbow angles are 2D projections in the image, shown for side-on shots only, and are not anatomical joint angles.";
}

function renderSquat(r, seek) {
  const s = r.summary, side = s.selected_side;
  $("tiles").innerHTML = [
    tile("Completed reps", s.completed_reps),
    tile("Incomplete attempts", s.incomplete_attempts),
    tile("Uncertain intervals", s.uncertain_attempts),
    tile("Tracked side", side || "—", side ? `${fmt(100 * s.leg_coverage[side], 0)}% leg coverage` : ""),
  ].join("");
  $("timeline-title").textContent = "Rep timeline";
  const landmark = (a, name, label) => {
    const frame = a[`source_frame_${name}`];
    if (frame == null) return `<span class="muted">${label} —</span>`;
    // Seek inside the annotated frame, including for fractional-frame trim starts.
    const time = (frame + 0.5) / r.meta.analysis_video.fps;
    return `<button type="button" class="landmark" data-t="${time}">${label} ${fmt(a[name + "_s"], 2, " s")}</button>`;
  };
  $("timeline").innerHTML = r.reps.map((a) => {
    const title = a.rep == null ? `Attempt ${a.attempt}` : `Rep ${a.rep}`;
    const depth = { above: "Hip above knee", "roughly level": "Hip roughly level with knee", below: "Hip below knee" };
    return `<article class="rep-event">
      <div class="rep-head"><button type="button" class="rep-play" data-t="${a.review_start_s}" aria-label="Play ${title}">▶ ${title}</button>
        <span class="pill ${a.rep == null ? "unknown" : "completed"}">${esc(a.status)}</span></div>
      ${a.reason ? `<p>${esc(a.reason)}</p>` : ""}
      <div class="landmarks">${landmark(a, "start", "Start")}${landmark(a, "bottom", "Bottom")}${landmark(a, "end", "End")}</div>
      <p>Descent ${fmt(a.descent_s, 2, " s")} · Ascent ${fmt(a.ascent_s, 2, " s")}</p>
      <p>Peak projected knee flexion ${fmt(a.peak_knee_flexion_projected_deg, 0, "°")} · Torso lean at bottom ${fmt(a.torso_lean_at_bottom_deg, 0, "°")}</p>
      <p><strong>2D depth estimate</strong> · ${depth[a.depth_2d] || "Unknown"}</p>
    </article>`;
  }).join("") || '<p class="rep-event muted">No squat attempts were detected.</p>';
  $("timeline").querySelectorAll("button[data-t]").forEach((el) => {
    el.addEventListener("click", () => seek(Number(el.dataset.t), 0, el.classList.contains("rep-play")));
  });
  $("charts").replaceChildren();
  $("limits").textContent = "Times refer to the original upload. Select a rep to play it, or a timestamp to pause on that frame. " +
    "Depth compares the visible hip and knee keypoints at the bottom. Knee flexion and torso lean from vertical are side-view projections. " +
    "A dash means the landmark or measurement was unavailable.";
}

function renderRunning(r, file, seek) {
  const s = r.summary;
  const conf = { ok: "", low: "low confidence", none: "not measurable" }[s.cadence_confidence];
  const knee = (f) => s.knee_flexion_at_strike_deg[f] ? `${fmt(s.knee_flexion_at_strike_deg[f].mean, 1)}°` : "—";
  $("tiles").innerHTML = [
    tile("Cadence", s.cadence_spm === null ? "—" : `${fmt(s.cadence_spm, 1)}<small> spm</small>`, `${s.strides_measured} strides${conf ? ` · ${conf}` : ""}`, s.cadence_confidence === "ok" ? "" : "unknown"),
    tile("Foot strikes", s.steps, `${s.steps_per_foot.left} left · ${s.steps_per_foot.right} right`),
    tile("Step time", fmt(s.mean_step_seconds, 3, " s")),
    tile("Stride time", fmt(s.mean_stride_seconds, 3, " s"), `variability ${fmt((s.stride_seconds_sd ?? NaN) * 1000, 0, " ms")}`),
    tile("Contact (proxy)", `${fmt(s.contact_seconds.left, 2)} / ${fmt(s.contact_seconds.right, 2)}`, "left / right, s"),
    tile("Knee near strike", `${knee("left")} / ${knee("right")}`, "projected 2D, left / right"),
  ].join("");
  $("timeline-title").textContent = "Step timeline";
  $("timeline").innerHTML = `<table class="steps"><thead><tr><th>#</th><th>foot</th><th>strike s</th><th>step s</th><th>stride s</th><th>contact s</th><th>knee °</th></tr></thead><tbody>${
    r.steps.map((st) => `<tr data-t="${st.time}"><td>${st.number}</td><td><span class="pill ${st.foot}">${st.foot}</span></td><td>${fmt(st.time, 2)}</td><td>${fmt(st.step_interval, 3)}</td><td>${fmt(st.stride_seconds, 3)}</td><td>${fmt(st.contact_seconds, 3)}</td><td>${fmt(st.knee_flexion_deg, 1)}</td></tr>`).join("")
  }</tbody></table><p class="muted small" style="padding:6px 10px">— = not measured (cut off by a tracking gap or the clip end, or not fully observed).</p>`;
  $("timeline").querySelectorAll("tr[data-t]").forEach((el) => el.addEventListener("click", () => seek(parseFloat(el.dataset.t))));
  $("charts").innerHTML = `<figure class="wide">${clearanceSvg(r)}<figcaption>Ankle clearance in leg lengths. Triangles mark detected strikes; grey bands are tracking gaps, which are left as gaps rather than interpolated. Click to jump the video.</figcaption></figure>`;
  const svg = $("charts").querySelector("svg");
  svg.addEventListener("click", (e) => {
    const box = svg.getBoundingClientRect(), x = (e.clientX - box.left) / box.width * 1000;
    const t = svg._t0 + (x - 50) / 930 * (svg._t1 - svg._t0);
    seek(t + 0.3);
  });
  $("limits").innerHTML = "<strong>Limits.</strong> One runner, filmed from the side with both ankles visible. Cadence = 120 ÷ mean same-foot stride time. " +
    "COCO-17 keypoints have no heel or toe, so foot-strike type (heel/forefoot), pronation and treadmill speed are not measured. " +
    "Contact time is time under a clearance threshold (a proxy). Knee angles are projected into the image plane and are not anatomical measurements.";
}

function clearanceSvg(r) {
  const c = r.chart, W = 1000, H = 260, L = 50, R = 20, T = 14, B = 30;
  const t0 = c.time[0], t1 = c.time[c.time.length - 1];
  const all = [...c.clearance.left, ...c.clearance.right].filter((v) => v !== null);
  const ymax = Math.max(0.1, ...all) * 1.08;
  const X = (t) => L + (t - t0) / (t1 - t0 || 1) * (W - L - R);
  const Y = (v) => T + (1 - v / ymax) * (H - T - B);
  const line = (vals) => { let d = "", pen = false; vals.forEach((v, i) => { if (v === null) { pen = false; return; } d += `${pen ? "L" : "M"}${X(c.time[i]).toFixed(1)},${Y(v).toFixed(1)}`; pen = true; }); return d; };
  const gaps = (vals) => { let out = "", start = null; vals.forEach((v, i) => { if (v === null && start === null) start = i; if ((v !== null || i === vals.length - 1) && start !== null) { out += `<rect x="${X(c.time[start])}" y="${T}" width="${Math.max(2, X(c.time[i]) - X(c.time[start]))}" height="${H - T - B}" fill="#adb5bd" opacity=".25"/>`; start = null; } }); return out; };
  const ticks = []; for (let t = Math.ceil(t0); t <= t1; t += Math.max(1, Math.round((t1 - t0) / 10))) ticks.push(`<line x1="${X(t)}" x2="${X(t)}" y1="${H - B}" y2="${H - B + 4}" stroke="currentColor"/><text x="${X(t)}" y="${H - 8}" font-size="11" text-anchor="middle" fill="currentColor">${t}s</text>`);
  const marks = r.steps.map((s) => `<path d="M${X(s.time) - 5},${Y(c.contact_level) - 9}h10l-5,9z" fill="var(--${s.foot})"/>`).join("");
  const svg = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Ankle clearance chart" style="color:var(--muted);cursor:pointer">
    ${gaps(c.clearance.left)}${gaps(c.clearance.right)}
    <line x1="${L}" x2="${W - R}" y1="${Y(c.contact_level)}" y2="${Y(c.contact_level)}" stroke="currentColor" stroke-dasharray="4 4"/>
    <path d="${line(c.clearance.left)}" fill="none" stroke="var(--left)" stroke-width="1.6"/>
    <path d="${line(c.clearance.right)}" fill="none" stroke="var(--right)" stroke-width="1.6"/>
    ${marks}<line x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}" stroke="currentColor"/>${ticks.join("")}
    <text x="${L}" y="${T + 10}" font-size="11" fill="var(--left)">left ankle</text><text x="${L + 70}" y="${T + 10}" font-size="11" fill="var(--right)">right ankle</text>
    <text x="${W - R}" y="${Y(c.contact_level) - 4}" font-size="11" text-anchor="end" fill="currentColor">strike threshold</text></svg>`;
  setTimeout(() => { const el = $("charts").querySelector("svg"); if (el) { el._t0 = t0; el._t1 = t1; } });
  return svg;
}

boot();
