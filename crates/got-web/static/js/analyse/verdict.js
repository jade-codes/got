// analyse/verdict.js — verdict banner + score displays

import { scoreColour, trustColour } from '../shared/colors.js';

export function updateScore(s) {
  const scoreValue = document.getElementById('scoreValue');
  const scoreLabel = document.getElementById('scoreLabel');
  scoreValue.textContent = s.toFixed(2);
  scoreValue.style.color = scoreColour(s);
  if (s >= 0.9) scoreLabel.textContent = 'COHERENT';
  else if (s >= 0.7) scoreLabel.textContent = 'DRIFTING';
  else if (s >= 0.4) scoreLabel.textContent = 'INCOHERENT';
  else scoreLabel.textContent = 'CONTRADICTORY';
  scoreLabel.style.color = scoreColour(s);
}

export function updateTrust(t) {
  const trustValue = document.getElementById('trustValue');
  const trustLabel = document.getElementById('trustLabel');
  trustValue.textContent = t.toFixed(2);
  trustValue.style.color = trustColour(t);
  if (t >= 0.8) trustLabel.textContent = 'TRUSTWORTHY';
  else if (t >= 0.5) trustLabel.textContent = 'UNCERTAIN';
  else if (t >= 0.2) trustLabel.textContent = 'SUSPICIOUS';
  else trustLabel.textContent = 'UNTRUSTWORTHY';
  trustLabel.style.color = trustColour(t);
}

export function renderVerdict(analysis) {
  const banner = document.getElementById('verdictBanner');
  const icon = document.getElementById('verdictIcon');
  const label = document.getElementById('verdictLabel');
  const summary = document.getElementById('verdictSummary');
  const metrics = document.getElementById('verdictMetrics');

  if (!analysis || !analysis.assessment) {
    banner.classList.remove('show');
    return;
  }

  const a = analysis.assessment;
  const v = a.verdict;

  const icons = { manipulative: '\u{1F6A8}', inconsistent: '\u26A0\uFE0F', drifting: '\u{1F4C9}', coherent: '\u2705' };
  icon.textContent = icons[v] || '\u2753';
  label.textContent = v.toUpperCase();
  summary.textContent = a.summary;

  let metricsHtml = 'Coherence: ' + (a.final_coherence * 100).toFixed(0) + '% | ' +
    'Trust: ' + (a.final_trust * 100).toFixed(0) + '% | ' +
    'Influence: ' + (a.influence_score * 100).toFixed(0) + '%' +
    ' | Msg Coherence: ' + ((a.mean_message_coherence || 1) * 100).toFixed(0) + '%' +
    ' | Convergence: ' + ((a.final_convergence || 0) * 100).toFixed(0) + '%';

  if (analysis.speaker_summary) {
    analysis.speaker_summary.forEach(s => {
      metricsHtml += ' | ' + s.speaker + ' drift: ' + (s.semantic_drift * 100).toFixed(0) + '%';
    });
  }
  metrics.innerHTML = metricsHtml;

  banner.className = 'verdict-banner show verdict-' + v;
}
