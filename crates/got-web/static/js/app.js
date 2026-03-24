// app.js — unified application: live chat + demo replay + all visualizations
//
// Single flow: every message (typed or replayed) goes through:
//   1. embed (if typed) or use pre-computed embedding (if demo)
//   2. proxy observe → detected values + deviation
//   3. accumulate for coherence analysis → contradictions, pairwise, etc.
//   4. update all visualizations

import { fetchDemoConversation, analyseConversation, embedText,
         createProxySession, proxyObserve, proxyManifold, proxySnapshot,
         chatWithModel, fetchCoherence, fetchCollapse, fetchCompare } from './api.js';
import { scoreColour, trustColour } from './shared/colors.js';
import { esc, getTermList, findPair } from './shared/utils.js';
import { renderTimeline, updateTimelineHighlight } from './analyse/timeline.js';
import { renderVerdict, updateScore, updateTrust } from './analyse/verdict.js';
import { renderContradictions, renderRedundancies } from './analyse/viz-contradictions.js';
import { renderChordDiagram } from './analyse/viz-chord.js';
import { renderHeatmap } from './analyse/viz-heatmap.js';
import { renderSphere } from './analyse/viz-sphere.js';
import { renderManifoldTerrain } from './analyse/viz-manifold.js';

// ---- State ----
let sessionId = null;
let analysis = null;          // latest coherence analysis result
let currentTurn = -1;
let playInterval = null;
let observationCount = 0;
let deviationHistory = [];
let attestations = [];
let allMessages = [];         // accumulated messages [{speaker, text, embedding}]
let speakerMap = {};
let latestSnapshot = null;    // most recent snapshot response (manifold data)

// ---- LLM settings ----
let llmProvider, llmApiKey, llmModel, llmBaseUrl;
let conversationHistory = []; // [{role, content}] for LLM context

// ---- DOM refs (grabbed once in init) ----
let chatBody, chatInput, btnSend, btnLoadDemo, btnPrev, btnNext, btnPlay;
let spinner;

// ---- Session management ----

async function ensureSession() {
  if (sessionId) return;
  try {
    const result = await createProxySession('interactive');
    sessionId = result.session_id;
    observationCount = 0;
    deviationHistory = [];
    attestations = [];

    document.getElementById('sessionDot').classList.add('active');
    document.getElementById('sessionInfo').textContent = 'Session: ' + sessionId.substring(0, 12) + '...';
  } catch (err) {
    console.error('Failed to create session:', err);
  }
}

// ---- Message sending (live input) ----

function getLLMSettings() {
  return {
    provider: llmProvider.value,
    apiKey: llmApiKey.value.trim(),
    model: llmModel.value.trim(),
    baseUrl: llmBaseUrl.value.trim() || undefined,
  };
}

function saveLLMSettings() {
  const s = getLLMSettings();
  localStorage.setItem('got_llm', JSON.stringify({
    provider: s.provider, model: s.model, baseUrl: s.baseUrl || '',
    // Never persist API key
  }));
}

function loadLLMSettings() {
  try {
    const raw = localStorage.getItem('got_llm');
    if (!raw) return; // no saved settings — use HTML defaults
    const s = JSON.parse(raw);
    if (s.provider) llmProvider.value = s.provider;
    if (s.model) llmModel.value = s.model;
    if (s.baseUrl) llmBaseUrl.value = s.baseUrl;
  } catch (_) {}
}

async function handleSend() {
  const text = chatInput.value.trim();
  if (!text) return;

  const settings = getLLMSettings();
  const hasLLM = settings.provider === 'ollama' || settings.apiKey.length > 0;

  chatInput.value = '';
  chatInput.disabled = true;
  btnSend.disabled = true;
  saveLLMSettings();

  try {
    await ensureSession();

    // Show user message in chat, observe as "user" (context only, no deviation)
    const userEmbedResult = await embedText(text);
    const userObs = await proxyObserve(sessionId, userEmbedResult.embedding, 'user');
    allMessages.push({ speaker: 'user', text, embedding: userEmbedResult.embedding });
    addMessageToChat(text, 'user', userObs.detected_values, userEmbedResult, 'You');

    if (hasLLM) {
      // Send to LLM
      conversationHistory.push({ role: 'user', content: text });

      let aiText;
      try {
        // Ollama uses OpenAI-compatible API
        const provider = settings.provider === 'ollama' ? 'openai' : settings.provider;
        const apiKey = settings.apiKey || 'ollama';  // Ollama doesn't need a real key
        const chatResult = await chatWithModel(
          provider, apiKey, settings.model,
          conversationHistory, settings.baseUrl
        );
        aiText = chatResult.response;
      } catch (err) {
        // Show error as system message
        addSystemMessage('LLM error: ' + err.message);
        chatInput.disabled = false;
        btnSend.disabled = false;
        chatInput.focus();
        return;
      }

      conversationHistory.push({ role: 'assistant', content: aiText });

      // Embed the AI response and observe through proxy as "assistant"
      const embedResult = await embedText(aiText);
      const obsResult = await proxyObserve(sessionId, embedResult.embedding, 'assistant');
      observationCount = obsResult.observation_count;

      // Show AI response with detected values
      allMessages.push({ speaker: 'assistant', text: aiText, embedding: embedResult.embedding });
      addMessageToChat(aiText, 'assistant', obsResult.detected_values, embedResult,
        settings.model || 'AI');

      // Update deviation
      if (obsResult.deviation) {
        deviationHistory.push(obsResult.deviation);
      }
      updateDeviationDisplay(obsResult.deviation || null);
      updateTopValues(obsResult.detected_values);

    } else {
      // No LLM configured — user message already observed above as "user"
      const obsResult = userObs;
      observationCount = obsResult.observation_count;

      // Update the user message with detected values
      updateLastMessageValues(obsResult.detected_values, userEmbedResult);

      if (obsResult.deviation) {
        deviationHistory.push(obsResult.deviation);
      }
      updateDeviationDisplay(obsResult.deviation || null);
      updateTopValues(obsResult.detected_values);
    }

    // Run coherence analysis
    await runCoherenceAnalysis();
    updateDeviationTimeline();
    updateCoherenceTab();

  } catch (err) {
    console.error('Send failed:', err);
    addSystemMessage('Error: ' + err.message);
  } finally {
    chatInput.disabled = false;
    btnSend.disabled = false;
    chatInput.focus();
  }
}

function addSystemMessage(text) {
  const div = document.createElement('div');
  div.className = 'message system-msg';
  div.innerHTML = '<div class="msg-bubble" style="background:rgba(248,81,73,0.08);border-color:rgba(248,81,73,0.25);color:#f85149;font-size:12px;">'
    + esc(text) + '</div>';
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

function updateLastMessageValues(detectedValues, embedInfo) {
  const msgs = chatBody.querySelectorAll('.message');
  if (msgs.length === 0) return;
  const last = msgs[msgs.length - 1];
  const bubble = last.querySelector('.msg-bubble');
  if (!bubble) return;

  if (detectedValues && detectedValues.length > 0) {
    let html = '<div class="msg-values">';
    detectedValues.forEach(v => {
      const score = v.score !== undefined ? v.score : v.cos_phi;
      const cls = score < 0 ? 'negative' : 'new';
      const sign = score >= 0 ? '+' : '';
      html += '<span class="value-chip ' + cls + '" title="z = ' + score.toFixed(3) + '">'
        + esc(v.term) + ' <small>' + sign + score.toFixed(2) + '</small></span>';
    });
    html += '</div>';
    bubble.insertAdjacentHTML('beforeend', html);
  }
  if (embedInfo) {
    bubble.insertAdjacentHTML('beforeend',
      '<div class="msg-tokens">' + embedInfo.matched_tokens + '/' + embedInfo.total_tokens + ' tokens matched</div>');
  }
}

// ---- Demo loading (replays through the same pipeline) ----

async function loadDemo() {
  spinner.classList.add('show');
  try {
    await ensureSession();

    const conversation = await fetchDemoConversation();
    speakerMap = {};
    conversation.participants.forEach(p => { speakerMap[p.id] = p; });

    document.getElementById('chatHeader').textContent = conversation.title || 'Conversation';

    // Update mode banner
    // Clear state
    allMessages = [];
    deviationHistory = [];
    chatBody.innerHTML = '';
    document.getElementById('emptyChat')?.remove();

    // Replay each message through the proxy with speaker attribution
    for (const msg of conversation.messages) {
      // Map demo speaker IDs: "user" stays "user", anything else is the model
      const proxySpeaker = msg.speaker === 'user' ? 'user' : 'assistant';
      const obsResult = await proxyObserve(sessionId, msg.embedding, proxySpeaker);
      observationCount = obsResult.observation_count;

      allMessages.push({ speaker: msg.speaker, text: msg.text, embedding: msg.embedding });

      const speakerInfo = speakerMap[msg.speaker];
      const label = speakerInfo ? speakerInfo.label : msg.speaker;
      addMessageToChat(msg.text, msg.speaker, obsResult.detected_values, null, label);

      if (obsResult.deviation) {
        deviationHistory.push(obsResult.deviation);
      }
    }

    // Run full coherence analysis
    await runCoherenceAnalysis();

    // Render timeline + verdict from coherence analysis
    if (analysis) {
      renderTimeline(analysis, selectTurn);
      renderVerdict(analysis);
      selectTurn(analysis.turns.length - 1);
    }

    // Show latest deviation
    const lastDev = deviationHistory.length > 0 ? deviationHistory[deviationHistory.length - 1] : null;
    updateDeviationDisplay(lastDev);
    if (analysis && analysis.turns.length > 0) {
      const lastTurn = analysis.turns[analysis.turns.length - 1];
      updateTopValuesFromTurn(lastTurn);
    }
    updateDeviationTimeline();
    updateCoherenceTab();

  } catch (err) {
    console.error('Failed to load demo:', err);
  } finally {
    spinner.classList.remove('show');
  }
}

// ---- Coherence analysis (runs on accumulated messages) ----

async function runCoherenceAnalysis() {
  if (allMessages.length === 0) return;

  try {
    analysis = await analyseConversation(allMessages);

    if (analysis && analysis.turns.length > 0) {
      renderTimeline(analysis, selectTurn);
      renderVerdict(analysis);
      selectTurn(analysis.turns.length - 1);
    }

    // Auto-fetch manifold data (non-blocking)
    if (sessionId && observationCount >= 5) {
      fetchManifold();
    }
  } catch (err) {
    console.error('Coherence analysis failed:', err);
  }
}

// ---- Turn selection (for coherence viz) ----

function selectTurn(idx) {
  if (!analysis || idx < 0 || idx >= analysis.turns.length) return;
  currentTurn = idx;

  // Highlight messages
  document.querySelectorAll('.message').forEach(el => {
    const t = parseInt(el.dataset.turn);
    el.classList.toggle('selected', t === idx);
    el.classList.toggle('dimmed', t > idx);
  });

  const turn = analysis.turns[idx];
  const sel = document.querySelector('.message.selected');
  if (sel) sel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  updateScore(turn.coherence_score);
  // Composite trust: server coherence-trust × deviation conformity × manifold health
  const compositeTrust = computeCompositeTrust(turn.trust_score, latestDeviation);
  updateTrust(compositeTrust);
  updateTermLegend(turn);
  updateTimelineHighlight(idx, analysis);

  // Update coherence tabs
  renderContradictions(turn);
  renderRedundancies(turn);
  renderChordDiagram(turn);
  renderHeatmap(turn);
  const prevTurn = idx > 0 ? analysis.turns[idx - 1] : null;
  renderSphere(turn, prevTurn, latestSnapshot);
}

// ---- Chat rendering ----

function addMessageToChat(text, speaker, detectedValues, embedInfo, displayLabel) {
  const emptyChat = document.getElementById('emptyChat');
  if (emptyChat) emptyChat.style.display = 'none';

  const speakers = Object.keys(speakerMap);
  const speakerA = speakers[0] || 'user';
  const isA = speaker === speakerA;
  const turnIdx = allMessages.length - 1;

  const div = document.createElement('div');
  div.className = 'message msg-speaker-' + (isA ? 'a' : 'b');
  div.dataset.turn = turnIdx;

  const label = displayLabel || speaker;

  let html = '<div class="msg-bubble">';
  html += '<div class="msg-meta">';
  html += '<span class="msg-speaker">' + esc(label) + '</span>';
  html += '<span class="msg-turn">Turn ' + turnIdx + '</span>';
  html += '</div>';
  html += '<div class="msg-text">' + esc(text) + '</div>';

  if (detectedValues && detectedValues.length > 0) {
    html += '<div class="msg-values">';
    detectedValues.forEach(v => {
      const score = v.score !== undefined ? v.score : v.cos_phi;
      const cls = score < 0 ? 'negative' : 'new';
      const sign = score >= 0 ? '+' : '';
      html += '<span class="value-chip ' + cls + '" title="z = ' + score.toFixed(3) + '">'
        + esc(v.term) + ' <small>' + sign + score.toFixed(2) + '</small></span>';
    });
    html += '</div>';
  }

  if (embedInfo) {
    html += '<div class="msg-tokens">' + embedInfo.matched_tokens + '/' + embedInfo.total_tokens + ' tokens matched</div>';
  }

  html += '</div>';
  div.innerHTML = html;
  div.addEventListener('click', () => selectTurn(turnIdx));
  chatBody.appendChild(div);
  chatBody.scrollTop = chatBody.scrollHeight;
}

// ---- Term legend ----

function updateTermLegend(turn) {
  const termLegend = document.getElementById('termLegend');
  termLegend.innerHTML = '';
  const introduced = new Set(turn.values_introduced);
  turn.cumulative_values.forEach(v => {
    const chip = document.createElement('span');
    chip.className = 'legend-chip' + (introduced.has(v) ? ' active' : '');
    chip.textContent = v;
    termLegend.appendChild(chip);
  });
}

// ---- Composite trust ----
// Trust = coherence_stability × baseline_conformity × manifold_health
// Each factor in [0, 1]. Any one going to zero tanks trust.

let latestDeviation = null; // most recent deviation from proxy

function computeCompositeTrust(coherenceTrust, deviation) {
  // Factor 1: coherence stability (from server — coherence × drift penalty)
  const coherenceFactor = coherenceTrust;

  // Factor 2: baseline conformity (1 - deviation severity)
  let baselineFactor = 1.0;
  if (deviation && deviation.baseline_sufficient) {
    baselineFactor = Math.max(0, 1.0 - deviation.combined_score);
  }

  // Factor 3: manifold health (1 = on-manifold, 0 = off-manifold)
  let manifoldFactor = 1.0;
  if (deviation && deviation.manifold_density_score !== undefined) {
    manifoldFactor = Math.max(0, 1.0 - deviation.manifold_density_score);
  }

  return (coherenceFactor * baselineFactor * manifoldFactor);
}

// ---- Deviation display ----

function updateManifoldBadge(deviation) {
  const badge = document.getElementById('manifoldValue');
  const label = document.getElementById('manifoldLabel');
  if (!deviation || deviation.manifold_density_score === undefined) {
    badge.textContent = '--';
    badge.style.color = '#8b949e';
    label.textContent = 'MANIFOLD';
    return;
  }
  // Continuous: show the deviation combined_score as a manifold health indicator
  // Lower deviation = healthier
  const health = Math.max(0, 1.0 - deviation.combined_score);
  badge.textContent = health.toFixed(2);
  badge.style.color = health >= 0.7 ? '#3fb950' : health >= 0.4 ? '#d29922' : '#f85149';
  label.textContent = health >= 0.7 ? 'HEALTHY' : health >= 0.4 ? 'DRIFTING' : 'ANOMALOUS';
}

function updateDeviationDisplay(deviation) {
  const badge = document.getElementById('deviationValue');
  const verdict = document.getElementById('deviationVerdict');
  const obsEl = document.getElementById('obsCount');
  const baselineEl = document.getElementById('baselineProgress');

  obsEl.textContent = observationCount;
  if (deviation) latestDeviation = deviation;
  updateManifoldBadge(deviation);

  if (!deviation) {
    badge.textContent = '--';
    badge.style.color = '#8b949e';
    verdict.textContent = observationCount > 0 ? 'Building Baseline' : 'No Session';
    verdict.className = 'verdict-pill building';
    if (observationCount > 0 && observationCount < 5) {
      const pct = Math.min(100, (observationCount / 5) * 100);
      baselineEl.innerHTML = observationCount + '/5 ' +
        '<span class="progress-bar"><span class="progress-fill" style="width:' + pct + '%"></span></span>';
      baselineEl.style.display = '';
    } else {
      baselineEl.style.display = 'none';
    }
    updateSignals(null);
    return;
  }

  baselineEl.style.display = 'none';
  badge.textContent = deviation.combined_score.toFixed(2);
  const vc = deviation.verdict === 'within_baseline' ? 'within'
    : deviation.verdict === 'drifting' ? 'drifting' : 'deviated';
  badge.style.color = vc === 'within' ? '#3fb950' : vc === 'drifting' ? '#d29922' : '#f85149';
  verdict.textContent = vc === 'within' ? 'Within Baseline' : vc === 'drifting' ? 'Drifting' : 'Deviated';
  verdict.className = 'verdict-pill ' + vc;
  updateSignals(deviation);
}

function updateSignals(deviation) {
  const panel = document.getElementById('tab-deviation');
  if (!deviation) {
    panel.innerHTML = '<div class="empty-state"><p>Waiting for baseline (5 model responses required)</p></div>';
    return;
  }

  const signals = [
    { name: 'Term Z-Score Shift', value: deviation.term_score },
    { name: 'Profile Cosine Drift', value: Math.min(1, deviation.profile_drift / 2) },
    { name: 'Pairwise Disruption', value: deviation.relationship_score },
    { name: 'Manifold Density', value: deviation.manifold_density_score || 0 },
  ];

  let html = '';
  signals.forEach(s => {
    const pct = Math.min(100, s.value * 100);
    const col = s.value < 0.3 ? 'green' : s.value < 0.6 ? 'yellow' : 'red';
    const clr = col === 'green' ? '#3fb950' : col === 'yellow' ? '#d29922' : '#f85149';
    html += '<div class="signal-card">' +
      '<div class="signal-header"><span class="signal-name">' + s.name + '</span>' +
      '<span class="signal-value" style="color:' + clr + '">' + s.value.toFixed(3) + '</span></div>' +
      '<div class="signal-bar"><div class="signal-fill ' + col + '" style="width:' + pct + '%"></div></div></div>';
  });

  const cs = deviation.combined_score;
  const cc = cs < 0.3 ? 'green' : cs < 0.6 ? 'yellow' : 'red';
  const ccl = cc === 'green' ? '#3fb950' : cc === 'yellow' ? '#d29922' : '#f85149';
  html += '<div class="signal-card">' +
    '<div class="signal-header"><span class="signal-name">Combined Score</span>' +
    '<span class="signal-value" style="color:' + ccl + '">' + cs.toFixed(3) + '</span></div>' +
    '<div class="signal-bar"><div class="signal-fill ' + cc + '" style="width:' + Math.min(100, cs * 100) + '%"></div></div></div>';

  panel.innerHTML = html;
}

// ---- Top values display ----

function updateTopValues(detectedValues) {
  const panel = document.getElementById('tab-values');
  if (!detectedValues || detectedValues.length === 0) {
    panel.innerHTML = '<div class="empty-state"><p>No values detected yet</p></div>';
    return;
  }
  renderValuesList(panel, detectedValues.map(v => ({ term: v.term, score: v.score })));
}

function updateTopValuesFromTurn(turn) {
  const panel = document.getElementById('tab-values');
  if (!turn.detected_values || turn.detected_values.length === 0) {
    panel.innerHTML = '<div class="empty-state"><p>No values detected</p></div>';
    return;
  }
  renderValuesList(panel, turn.detected_values.map(v => ({ term: v.term, score: v.cos_phi })));
}

function renderValuesList(panel, values) {
  const sorted = [...values].sort((a, b) => b.score - a.score);
  const maxScore = Math.max(...sorted.map(v => Math.abs(v.score)), 1);

  let html = '<ul class="top-values-list">';
  sorted.forEach(v => {
    const pct = Math.min(100, (Math.abs(v.score) / maxScore) * 100);
    html += '<li><span class="term-name">' + esc(v.term) + '</span>' +
      '<span class="term-score">' + (v.score >= 0 ? '+' : '') + v.score.toFixed(2) + '</span>' +
      '<span class="term-bar"><span class="term-bar-fill" style="width:' + pct + '%"></span></span></li>';
  });
  html += '</ul>';
  panel.innerHTML = html;
}

// ---- Deviation timeline chart ----

function updateDeviationTimeline() {
  // Nothing to do — deviation data is shown in signal cards and the
  // manifold badge. The timeline chart shows coherence + trust lines.
}

// ---- Manifold tab ----

async function fetchManifold() {
  if (!sessionId) return;
  const btn = document.getElementById('btnManifold');
  const status = document.getElementById('manifoldStatus');
  btn.disabled = true;
  status.textContent = 'Computing...';

  try {
    const result = await proxyManifold(sessionId);
    status.textContent = 'Attested #' + result.sequence_number +
      ' (' + result.observation_count + ' obs) ' +
      result.attestation_hash.substring(0, 12) + '...';
    latestSnapshot = result;
    renderManifoldTab(result);
    // Re-render sphere if a turn is selected (now with density data)
    if (analysis && currentTurn >= 0) {
      const turn = analysis.turns[currentTurn];
      const prevTurn = currentTurn > 0 ? analysis.turns[currentTurn - 1] : null;
      renderSphere(turn, prevTurn, latestSnapshot);
    }
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    btn.disabled = false;
  }
}

function renderManifoldTab(snapshot) {
  const panel = document.getElementById('manifoldContent');
  panel.innerHTML = '';

  if (!snapshot.manifold_density && !snapshot.manifold_curvature) {
    panel.innerHTML = '<div class="empty-state"><p>Not enough activations for manifold analysis (need 5+)</p></div>';
    return;
  }

  // Terrain visualization (uses per-turn pairwise distances for MDS positioning)
  if (analysis && currentTurn >= 0 && snapshot.term_densities && Object.keys(snapshot.term_densities).length >= 2) {
    const vizContainer = document.createElement('div');
    vizContainer.className = 'viz-container';
    panel.appendChild(vizContainer);
    renderManifoldTerrain(vizContainer, analysis.turns[currentTurn], snapshot);
  }

  // Summary metrics below the terrain
  let html = '<div class="manifold-metrics-row">';

  if (snapshot.manifold_density) {
    const d = snapshot.manifold_density;
    html += '<div class="manifold-section"><h3>Density</h3>';
    html += manifoldMetric('Intrinsic Dim', d.mean_intrinsic_dim.toFixed(2),
      '\u00B1 ' + d.std_intrinsic_dim.toFixed(2));
    html += manifoldMetric('Log-Density', d.mean_log_density.toFixed(3), '');
    html += manifoldMetric('Points', d.num_points, d.num_degenerate > 0 ? d.num_degenerate + ' degen' : '');
    html += '</div>';
  }

  if (snapshot.manifold_curvature) {
    const c = snapshot.manifold_curvature;
    html += '<div class="manifold-section"><h3>Curvature</h3>';
    html += manifoldMetric('Mean \u03BA', c.mean_curvature.toFixed(4),
      '\u00B1 ' + c.std_curvature.toFixed(4));
    const sign = c.mean_curvature > 0.001 ? 'positive'
      : c.mean_curvature < -0.001 ? 'negative' : 'flat';
    html += manifoldMetric('Shape', sign, '');
    html += manifoldMetric('Points', c.num_points, c.num_degenerate > 0 ? c.num_degenerate + ' degen' : '');
    html += '</div>';
  }

  html += '</div>';
  panel.insertAdjacentHTML('beforeend', html);
}

function manifoldMetric(label, value, detail) {
  return '<div class="manifold-metric">' +
    '<span class="metric-label">' + label + '</span>' +
    '<span class="metric-value">' + value + '</span>' +
    (detail ? '<span class="metric-detail">' + detail + '</span>' : '') +
    '</div>';
}

// ---- Coherence tab ----

let coherenceScores = []; // per-message C(h) scores

async function updateCoherenceTab() {
  if (allMessages.length < 2) return;

  // Build ordering from available terms: pair adjacent terms as constraints
  // Use the first few terms as dominant/subordinate pairs
  const terms = Object.keys(allMessages[0]?.embedding ? {} : {});
  // Simple approach: use embeddings from allMessages
  const embeddings = allMessages.map(m => m.embedding).filter(Boolean);
  if (embeddings.length === 0) return;

  // Build default ordering from available value terms (positive > negative pairs)
  const defaultPairs = [
    { dominant: 'honesty', subordinate: 'secrecy', label: 'honesty > secrecy' },
    { dominant: 'compassion', subordinate: 'cruelty', label: 'compassion > cruelty' },
    { dominant: 'fairness', subordinate: 'oppression', label: 'fairness > oppression' },
    { dominant: 'courage', subordinate: 'cowardice', label: 'courage > cowardice' },
    { dominant: 'transparency', subordinate: 'secrecy', label: 'transparency > secrecy' },
    { dominant: 'freedom', subordinate: 'oppression', label: 'freedom > oppression' },
  ];

  try {
    const result = await fetchCoherence(defaultPairs, embeddings);
    if (result.per_message) {
      coherenceScores = result.per_message;
      renderCoherenceChart(result);
      // Update score strip
      const badge = document.getElementById('scoreValue');
      if (badge && result.mean !== undefined) {
        badge.textContent = result.mean.toFixed(2);
      }
    }
  } catch (err) {
    console.error('Coherence scoring failed:', err);
  }
}

function renderCoherenceChart(result) {
  const chartEl = document.getElementById('coherenceChart');
  const summaryEl = document.getElementById('coherenceSummary');
  const emptyEl = document.getElementById('coherenceEmpty');

  if (!result.per_message || result.per_message.length === 0) return;
  emptyEl.style.display = 'none';
  chartEl.style.display = 'flex';
  summaryEl.style.display = 'block';

  // D3 line chart
  chartEl.innerHTML = '';
  const margin = { top: 20, right: 20, bottom: 30, left: 40 };
  const width = 500 - margin.left - margin.right;
  const height = 250 - margin.top - margin.bottom;

  const svg = d3.select(chartEl).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, result.per_message.length - 1]).range([0, width]);
  const y = d3.scaleLinear().domain([0, 1]).range([height, 0]);

  svg.append('g').attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x).ticks(Math.min(result.per_message.length, 10)).tickFormat(d => `${d}`))
    .selectAll('text').style('fill', '#8b949e');
  svg.append('g').call(d3.axisLeft(y).ticks(5))
    .selectAll('text').style('fill', '#8b949e');
  svg.selectAll('.domain, .tick line').style('stroke', '#30363d');

  // Reference line at 0.5
  svg.append('line').attr('class', 'coherence-ref-line')
    .attr('x1', 0).attr('x2', width).attr('y1', y(0.5)).attr('y2', y(0.5));

  // Line
  const line = d3.line().x((d, i) => x(i)).y(d => y(d));
  svg.append('path').datum(result.per_message)
    .attr('class', 'coherence-line').attr('d', line);

  // Dots
  const violatedPositions = new Set((result.violated || []).map(v => v.position));
  svg.selectAll('.coherence-dot').data(result.per_message).enter()
    .append('circle')
    .attr('class', (d, i) => 'coherence-dot' + (violatedPositions.has(i) ? ' violated' : ''))
    .attr('cx', (d, i) => x(i)).attr('cy', d => y(d)).attr('r', 4);

  // Summary card
  let html = '<div class="coherence-summary-card">';
  html += `<div style="display:flex;gap:24px;margin-bottom:8px;">`;
  html += `<div><span class="metric-label">Mean</span> <span class="metric-value">${result.mean.toFixed(3)}</span></div>`;
  html += `<div><span class="metric-label">Min</span> <span class="metric-value">${result.min.toFixed(3)}</span></div>`;
  html += `<div><span class="metric-label">Max</span> <span class="metric-value">${result.max.toFixed(3)}</span></div>`;
  html += '</div>';
  if (result.violated && result.violated.length > 0) {
    html += '<div style="margin-top:8px;font-size:12px;color:#f85149;">';
    html += '<strong>Violated constraints:</strong><ul style="margin:4px 0;padding-left:16px;">';
    result.violated.forEach(v => {
      html += `<li>Position ${v.position}: ${esc(v.label)} (margin: ${v.margin.toFixed(3)})</li>`;
    });
    html += '</ul></div>';
  }
  html += '</div>';
  summaryEl.innerHTML = html;
}

// ---- Collapse tab ----

async function fetchCollapseReport() {
  const btn = document.getElementById('btnCollapse');
  const status = document.getElementById('collapseStatus');
  const content = document.getElementById('collapseContent');
  btn.disabled = true;
  status.textContent = 'Computing...';

  try {
    const result = await fetchCollapse();
    renderCollapseChart(result, content);
    status.textContent = `dim_eff = ${result.dim_eff.toFixed(2)} / ${result.k}`;
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
    content.innerHTML = '<div class="empty-state"><p>Failed to compute collapse report</p></div>';
  } finally {
    btn.disabled = false;
  }
}

function renderCollapseChart(result, container) {
  container.innerHTML = '';

  // Ratio display
  const ratio = result.dim_eff_ratio;
  const assessCls = result.assessment.replace(/\s+/g, '-');
  let html = `<div style="text-align:center;margin-bottom:16px;">`;
  html += `<div class="collapse-ratio">${result.dim_eff.toFixed(2)} / ${result.k}.00 = ${(ratio * 100).toFixed(0)}%</div>`;
  html += `<span class="collapse-assessment ${assessCls}">${result.assessment}</span>`;
  html += '</div>';
  container.insertAdjacentHTML('beforeend', html);

  // D3 bar chart of eigenvalues
  const vizDiv = document.createElement('div');
  vizDiv.className = 'viz-container';
  container.appendChild(vizDiv);

  const margin = { top: 10, right: 20, bottom: 30, left: 50 };
  const width = 400 - margin.left - margin.right;
  const height = 200 - margin.top - margin.bottom;

  const svg = d3.select(vizDiv).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const eigenvalues = result.eigenvalues;
  const maxEv = Math.max(...eigenvalues.map(Math.abs), 0.01);

  const x = d3.scaleBand().domain(eigenvalues.map((_, i) => `λ${i}`)).range([0, width]).padding(0.3);
  const y = d3.scaleLinear().domain([0, maxEv]).range([height, 0]);

  const colorScale = d3.scaleSequential(d3.interpolateWarm).domain([0, maxEv]);

  svg.append('g').attr('transform', `translate(0,${height})`)
    .call(d3.axisBottom(x)).selectAll('text').style('fill', '#8b949e');
  svg.append('g').call(d3.axisLeft(y).ticks(5))
    .selectAll('text').style('fill', '#8b949e');
  svg.selectAll('.domain, .tick line').style('stroke', '#30363d');

  svg.selectAll('.collapse-bar').data(eigenvalues).enter()
    .append('rect').attr('class', 'collapse-bar')
    .attr('x', (d, i) => x(`λ${i}`)).attr('width', x.bandwidth())
    .attr('y', d => y(Math.max(0, d))).attr('height', d => height - y(Math.max(0, d)))
    .attr('fill', d => colorScale(Math.abs(d)));

  // Explanation
  container.insertAdjacentHTML('beforeend',
    '<p class="collapse-explanation">Effective dimensionality measures how many independent value dimensions ' +
    'the model\'s geometry actually uses. A ratio near 100% means all value directions are distinct. ' +
    'Near 0% means they\'ve collapsed together into fewer dimensions.</p>');
}

// ---- Compare tab ----

async function fetchCompareReport() {
  const pathInput = document.getElementById('compareGotuePath');
  const content = document.getElementById('compareContent');
  const path = pathInput.value.trim();
  if (!path) {
    content.innerHTML = '<div class="empty-state"><p>Enter a .gotue file path first</p></div>';
    return;
  }

  const btn = document.getElementById('btnCompare');
  btn.disabled = true;

  try {
    const result = await fetchCompare(path);
    renderCompareChart(result, content);
  } catch (err) {
    content.innerHTML = '<div class="empty-state"><p>Error: ' + esc(err.message || String(err)) + '</p></div>';
  } finally {
    btn.disabled = false;
  }
}

function renderCompareChart(result, container) {
  container.innerHTML = '';

  // Summary metrics
  let html = '<div style="display:flex;gap:24px;margin-bottom:16px;">';
  html += `<div class="manifold-metric"><span class="metric-label">Global Distance</span><span class="metric-value">${result.global_distance.toFixed(6)}</span></div>`;
  if (result.probe_projected_distance !== null) {
    html += `<div class="manifold-metric"><span class="metric-label">Probe-Projected</span><span class="metric-value">${result.probe_projected_distance.toFixed(6)}</span></div>`;
  }
  if (result.ratio !== null) {
    const ratioStr = result.ratio.toFixed(1) + 'x';
    const marker = result.ratio > 2.0 ? ' (value change exceeds global)' : '';
    html += `<div class="manifold-metric"><span class="metric-label">Ratio</span><span class="metric-value" style="color:${result.ratio > 2 ? '#f85149' : '#c9d1d9'}">${ratioStr}${marker}</span></div>`;
  }
  html += '</div>';
  container.insertAdjacentHTML('beforeend', html);

  if (!result.per_probe || result.per_probe.length === 0) return;

  // D3 horizontal bar chart of per-probe distances
  const vizDiv = document.createElement('div');
  vizDiv.className = 'viz-container';
  container.appendChild(vizDiv);

  const sorted = [...result.per_probe].sort((a, b) => b.distance - a.distance);
  const margin = { top: 10, right: 20, bottom: 10, left: 100 };
  const barH = 24;
  const width = 400 - margin.left - margin.right;
  const height = sorted.length * barH;

  const svg = d3.select(vizDiv).append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .append('g').attr('transform', `translate(${margin.left},${margin.top})`);

  const x = d3.scaleLinear().domain([0, Math.max(...sorted.map(d => d.distance), 0.01)]).range([0, width]);
  const y = d3.scaleBand().domain(sorted.map(d => d.label)).range([0, height]).padding(0.2);

  svg.selectAll('.compare-bar').data(sorted).enter()
    .append('rect')
    .attr('x', 0).attr('y', d => y(d.label)).attr('width', d => x(d.distance)).attr('height', y.bandwidth())
    .attr('fill', '#f78166').attr('rx', 3);

  svg.selectAll('.compare-label').data(sorted).enter()
    .append('text')
    .attr('x', -8).attr('y', d => y(d.label) + y.bandwidth() / 2)
    .attr('text-anchor', 'end').attr('dominant-baseline', 'middle')
    .style('fill', '#c9d1d9').style('font-size', '12px')
    .text(d => d.label);

  svg.selectAll('.compare-val').data(sorted).enter()
    .append('text')
    .attr('x', d => x(d.distance) + 6).attr('y', d => y(d.label) + y.bandwidth() / 2)
    .attr('dominant-baseline', 'middle')
    .style('fill', '#8b949e').style('font-size', '11px').style('font-family', 'monospace')
    .text(d => d.distance.toFixed(4));
}

// ---- Init ----

export function init() {
  chatBody = document.getElementById('chatBody');
  chatInput = document.getElementById('chatInput');
  btnSend = document.getElementById('btnSend');
  btnLoadDemo = document.getElementById('btnLoadDemo');
  btnPrev = document.getElementById('btnPrev');
  btnNext = document.getElementById('btnNext');
  btnPlay = document.getElementById('btnPlay');
  spinner = document.getElementById('spinner');

  // LLM settings
  llmProvider = document.getElementById('llmProvider');
  llmApiKey = document.getElementById('llmApiKey');
  llmModel = document.getElementById('llmModel');
  llmBaseUrl = document.getElementById('llmBaseUrl');
  loadLLMSettings();

  // Update defaults based on provider
  llmProvider.addEventListener('change', () => {
    const p = llmProvider.value;
    if (p === 'ollama') {
      llmModel.value = 'qwen3.5:9b';
      llmBaseUrl.value = 'http://localhost:11434/v1';
      llmApiKey.placeholder = 'Not needed for Ollama';
    } else if (p === 'anthropic') {
      llmModel.value = 'claude-sonnet-4-20250514';
      llmBaseUrl.value = '';
      llmApiKey.placeholder = 'API key';
    } else {
      llmModel.value = 'gpt-4o';
      llmBaseUrl.value = '';
      llmApiKey.placeholder = 'API key';
    }
    saveLLMSettings();
  });

  // Send message
  btnSend.addEventListener('click', handleSend);
  chatInput.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Load demo
  btnLoadDemo.addEventListener('click', loadDemo);

  // Manifold computation
  document.getElementById('btnManifold').addEventListener('click', fetchManifold);

  // Collapse computation
  document.getElementById('btnCollapse').addEventListener('click', fetchCollapseReport);

  // Compare computation
  document.getElementById('btnCompare').addEventListener('click', fetchCompareReport);

  // Navigation
  btnPrev.addEventListener('click', () => {
    if (analysis && currentTurn > 0) selectTurn(currentTurn - 1);
  });
  btnNext.addEventListener('click', () => {
    if (analysis && currentTurn < analysis.turns.length - 1) selectTurn(currentTurn + 1);
  });
  btnPlay.addEventListener('click', () => {
    if (playInterval) {
      clearInterval(playInterval);
      playInterval = null;
      btnPlay.innerHTML = '&#x25B6;';
      btnPlay.classList.remove('active');
    } else if (analysis) {
      if (currentTurn >= analysis.turns.length - 1) selectTurn(0);
      btnPlay.innerHTML = '&#x23F8;';
      btnPlay.classList.add('active');
      playInterval = setInterval(() => {
        if (currentTurn < analysis.turns.length - 1) {
          selectTurn(currentTurn + 1);
        } else {
          clearInterval(playInterval);
          playInterval = null;
          btnPlay.innerHTML = '&#x25B6;';
          btnPlay.classList.remove('active');
        }
      }, 1800);
    }
  });

  // Tab switching
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    });
  });

  // Keyboard navigation
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { btnPrev.click(); e.preventDefault(); }
    if (e.key === 'ArrowRight') { btnNext.click(); e.preventDefault(); }
    if (e.key === ' ') { btnPlay.click(); e.preventDefault(); }
  });
}
