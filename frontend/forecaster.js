// Forecaster — clean, bracket-first. Dependency-free; talks to the FastAPI backend.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const api = (p, o) => fetch(p, o).then((r) => { if (!r.ok) throw new Error(p + " " + r.status); return r.json(); });
  const pct = (x, dp) => (x == null ? "·" : (x * 100).toFixed(dp || 0) + "%");
  const esc = (s) => String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"];
  function niceDate(iso) { const [y, m, d] = iso.split("-").map(Number); return `${d} ${MONTHS[m - 1]} ${y}`; }

  // team -> flag code (flagcdn). England/Scotland use UK subdivision flags.
  const FLAG = {
    "Algeria":"dz","Argentina":"ar","Austria":"at","Jordan":"jo","Australia":"au",
    "Paraguay":"py","Turkey":"tr","United States":"us","Belgium":"be","Egypt":"eg",
    "Iran":"ir","New Zealand":"nz","Bosnia and Herzegovina":"ba","Canada":"ca","Qatar":"qa",
    "Switzerland":"ch","Brazil":"br","Haiti":"ht","Morocco":"ma","Scotland":"gb-sct",
    "Cape Verde":"cv","Saudi Arabia":"sa","Spain":"es","Uruguay":"uy","Colombia":"co",
    "DR Congo":"cd","Portugal":"pt","Uzbekistan":"uz","Croatia":"hr","England":"gb-eng",
    "Ghana":"gh","Panama":"pa","Curaçao":"cw","Ecuador":"ec","Germany":"de",
    "Ivory Coast":"ci","Czech Republic":"cz","Mexico":"mx","South Africa":"za","South Korea":"kr",
    "France":"fr","Iraq":"iq","Norway":"no","Senegal":"sn","Japan":"jp",
    "Netherlands":"nl","Sweden":"se","Tunisia":"tn",
  };
  function flag(name) {
    const c = FLAG[name];
    if (!c) return "";
    return `<img class="flag" src="https://flagcdn.com/h20/${c}.png" ` +
      `srcset="https://flagcdn.com/h40/${c}.png 2x" alt="" loading="lazy" ` +
      `onerror="this.style.display='none'">`;
  }
  const named = (t) => (t ? flag(t) + esc(t) : "TBD");

  const state = { comp: null, teams: [], bracket: null, view: "prediction" };

  async function init() {
    let comps;
    try { comps = await api("/forecaster/competitions"); }
    catch (e) { $("asof").textContent = "Forecaster artifacts not built. Run: python -m forecaster.build_artifacts"; return; }
    state.comp = comps[0].id;
    $("comp-title").textContent = comps[0].name + " Forecast";

    const t = await api("/forecaster/teams?competition=" + state.comp);
    state.teams = t.teams;
    fillSelect($("h-home"), state.teams, pick(state.teams, "Argentina", 0));
    fillSelect($("h-away"), state.teams, pick(state.teams, "Brazil", 1));
    ["h-home", "h-away"].forEach((id) => $(id).addEventListener("change", headToHead));

    $("bk-toggle").querySelectorAll("button").forEach((btn) => {
      btn.onclick = () => {
        state.view = btn.dataset.view;
        $("bk-toggle").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
        renderBracket();
      };
    });

    await Promise.all([loadBracketAndOdds(), headToHead(), loadGroups(), loadAccuracy()]);
    setInterval(loadBracketAndOdds, 120000); // refresh as games are played
  }

  const pick = (arr, want, i) => (arr.includes(want) ? want : arr[i] || arr[0]);
  function fillSelect(sel, teams, chosen) {
    sel.innerHTML = teams.map((t) => `<option${t === chosen ? " selected" : ""}>${esc(t)}</option>`).join("");
  }

  // ---- odds + bracket -----------------------------------------------------
  async function loadBracketAndOdds() {
    const [b, sim] = await Promise.all([
      api("/forecaster/bracket?competition=" + state.comp),
      api("/forecaster/simulation?competition=" + state.comp),
    ]);
    state.bracket = b;
    const n = b.settled_count;
    $("asof").textContent = `Updated through ${niceDate(b.as_of)}. ${n} knockout game${n === 1 ? "" : "s"} played so far.`;
    renderOdds(sim);
    renderBracket();
  }

  function renderOdds(sim) {
    const top = sim.teams.slice().sort((a, b) => b.champion - a.champion).slice(0, 10);
    const max = top[0].champion || 1;
    $("odds").innerHTML = top.map((r, i) => `
      <div class="odds-row ${i === 0 ? "lead-team" : ""}">
        <span class="odds-rank">${i + 1}</span>
        <span class="odds-name">${flag(r.team)}${esc(r.team)}</span>
        <span class="odds-track"><span class="odds-fill" style="width:${(r.champion / max) * 100}%"></span></span>
        <span class="odds-val">${pct(r.champion, r.champion < 0.1 ? 1 : 0)}</span>
      </div>`).join("");
  }

  // ---- bracket rendering --------------------------------------------------
  function bkRow(name, meta, cls) {
    return `<div class="bk-row ${cls || ""}">
      <span class="bk-name">${named(name)}</span>
      <span class="bk-meta">${meta || ""}</span></div>`;
  }

  function predMatch(m) {
    const aWin = m.winner === m.a, bWin = m.winner === m.b;
    return `<div class="bk-match"><div class="bk-box">
      ${bkRow(m.a, pct(m.prob_a), aWin ? "win" : "")}
      ${bkRow(m.b, pct(m.prob_b), bWin ? "win" : "")}
    </div></div>`;
  }

  function actualMatch(m) {
    if (!m.settled) {
      // not played yet — show the matchup (or TBD if a feeder isn't decided)
      const cls = (t) => (t ? "" : "tbd");
      return `<div class="bk-match"><div class="bk-box">
        ${bkRow(m.a, "", cls(m.a))}
        ${bkRow(m.b, "", cls(m.b))}
      </div></div>`;
    }
    const aWin = m.winner === m.a;
    const mark = m.correct ? `<span class="mark hit">✓</span>` : `<span class="mark miss">✗</span>`;
    const sa = `${m.score[0]} ${aWin ? mark : ""}`.trim();
    const sb = `${m.score[1]} ${!aWin ? mark : ""}`.trim();
    return `<div class="bk-match"><div class="bk-box">
      ${bkRow(m.a, sa, aWin ? "win" : "")}
      ${bkRow(m.b, sb, aWin ? "" : "win")}
    </div></div>`;
  }

  function renderBracket() {
    const b = state.bracket;
    if (!b) return;
    const data = b[state.view];
    const isPred = state.view === "prediction";

    if (isPred) {
      $("bk-note").innerHTML = `The model's single most likely outcome for all 31 knockout games, advanced to a predicted winner: <b>${esc(data.champion)}</b>.`;
    } else {
      const c = data.correct, d = data.decided;
      $("bk-note").innerHTML = d
        ? `So far <b>${c} of ${d}</b> finished knockout game${d === 1 ? "" : "s"} matched the model's prediction. A ✓ marks each correct one.`
        : `No knockout games have finished yet. This updates as they are played.`;
    }

    const renderMatch = isPred ? predMatch : actualMatch;
    const cols = data.rounds.map((rnd) => {
      const isFinal = rnd.round === "final";
      return `<div class="bk-col ${isFinal ? "pre-champ" : ""}">
        <div class="bk-head">${rnd.label}</div>
        <div class="bk-list">${rnd.matches.map(renderMatch).join("")}</div></div>`;
    }).join("");

    const champTbd = !data.champion;
    const champ = `<div class="bk-col champ-col">
      <div class="bk-head"></div>
      <div class="bk-list"><div class="bk-match"><div class="bk-champ ${champTbd ? "tbd" : ""}">
        <span class="cl">${isPred ? "Predicted champion" : "Champion"}</span>
        <span class="cn">${champTbd ? "TBD" : named(data.champion)}</span>
      </div></div></div></div>`;

    $("bracket").innerHTML = cols + champ;
  }

  // ---- head to head -------------------------------------------------------
  async function headToHead() {
    const home = $("h-home").value, away = $("h-away").value;
    const r = await api("/forecaster/match", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ competition: state.comp, home, away, neutral: true }),
    });
    const bar = $("h2h-bar");
    bar.querySelector(".home").style.width = r.prob_home * 100 + "%";
    bar.querySelector(".draw").style.width = r.prob_draw * 100 + "%";
    bar.querySelector(".away").style.width = r.prob_away * 100 + "%";
    $("h-home-p").textContent = pct(r.prob_home);
    $("h-draw-p").textContent = pct(r.prob_draw);
    $("h-away-p").textContent = pct(r.prob_away);
    $("h-home-n").innerHTML = flag(r.home) + esc(r.home);
    $("h-away-n").innerHTML = flag(r.away) + esc(r.away);
    $("h-score").textContent = `${r.most_likely[0]}–${r.most_likely[1]}`;
  }

  // ---- groups -------------------------------------------------------------
  async function loadGroups() {
    const g = await api("/forecaster/groups?competition=" + state.comp);
    $("groups").innerHTML = Object.keys(g.groups).sort().map((L) => {
      const rows = g.groups[L].map((r) => {
        const cls = r.advanced ? "adv" : (r.played ? "elim" : "");
        const mark = r.played ? (r.advanced ? "✓" : "✗") : "·";
        const pred = r.forecast_advance == null ? "·" : pct(r.forecast_advance);
        return `<div class="grow ${cls}">
          <span class="gp">${r.position}</span>
          <span class="gname">${flag(r.team)}${esc(r.team)}</span>
          <span class="gpred">${pred}</span>
          <span class="gout">${mark}</span></div>`;
      }).join("");
      return `<div class="gcard"><h3>Group ${L}</h3>
        <div class="gcol-head"><span></span><span>Team</span><span>Pred.</span><span>Res.</span></div>
        ${rows}</div>`;
    }).join("");
  }

  // ---- accuracy -----------------------------------------------------------
  async function loadAccuracy() {
    let mt;
    try { mt = await api("/forecaster/metrics?competition=" + state.comp); } catch (e) { return; }
    if (!mt || !mt.calibration) return;
    renderCalib(mt.calibration.bins);
    const m = mt.model, c = mt.config;
    $("acc-text").innerHTML = `
      <p>The model doesn't just pick winners. It puts a <b>probability</b> on every
      result, and what matters is whether those probabilities are accurate. Across
      <b>${c.n_test.toLocaleString()}</b> real matches it had never seen, they were
      <b>well-calibrated</b>: when it said 70%, that happened about 70% of the time.</p>
      <p class="acc-foot">Outright-winner accuracy is a weak measure of a probabilistic
      model. In a knockout there are no draws, so a coin flip alone is correct about
      50% of the time. The relevant measure is the chart above: predicted probability
      versus how often the result occurred, tracking the dashed line. The calibration
      error is <b>${m.ece.toFixed(3)}</b> (0 is perfect), and it scores <b>${m.log_loss.toFixed(2)}</b>
      on log-loss versus ${mt.baseline.log_loss.toFixed(2)} for a naive baseline, where
      lower is better.</p>`;
    $("fc-foot").textContent =
      `Scoreline model: Dixon-Coles (bivariate Poisson), fit on ~49k international results. ` +
      `Tournament odds from 10,000 Monte Carlo simulations. The strength model is frozen before ` +
      `the tournament; results only decide who advances. Live results come from a public ` +
      `community dataset, which can lag actual results by a few hours.`;
  }

  function renderCalib(bins) {
    const W = 300, H = 300, pad = 34;
    const X = (p) => pad + p * (W - 2 * pad), Y = (p) => H - pad - p * (H - 2 * pad);
    const maxN = Math.max(...bins.map((b) => b.n), 1);
    const pts = bins.map((b) => `<circle cx="${X(b.p_pred).toFixed(1)}" cy="${Y(b.p_obs).toFixed(1)}" r="${(3 + 5 * Math.sqrt(b.n / maxN)).toFixed(1)}" class="cpt"/>`).join("");
    const path = bins.map((b, i) => (i ? "L" : "M") + X(b.p_pred).toFixed(1) + " " + Y(b.p_obs).toFixed(1)).join(" ");
    const grid = [0, 0.5, 1].map((v) =>
      `<line x1="${X(v)}" y1="${Y(0)}" x2="${X(v)}" y2="${Y(1)}" class="cax"/><line x1="${X(0)}" y1="${Y(v)}" x2="${X(1)}" y2="${Y(v)}" class="cax"/>`).join("");
    $("acc-chart").innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Calibration curve">
        <style>
          .cax{stroke:var(--border);stroke-width:1}
          .cdiag{stroke:var(--text-muted);stroke-width:1.4;stroke-dasharray:5 4;opacity:.7}
          .cline{fill:none;stroke:var(--go);stroke-width:2.4}
          .cpt{fill:var(--go);stroke:var(--surface);stroke-width:1.5}
          .clbl{fill:var(--text-muted);font-size:11px}
        </style>
        ${grid}
        <line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" class="cdiag"/>
        <path d="${path}" class="cline"/>${pts}
        <text x="${X(0.5)}" y="${H - 6}" text-anchor="middle" class="clbl">Prediction</text>
        <text x="12" y="${Y(0.5)}" text-anchor="middle" class="clbl" transform="rotate(-90 12 ${Y(0.5)})">Reality</text>
      </svg>`;
  }

  document.addEventListener("DOMContentLoaded", init);
})();
