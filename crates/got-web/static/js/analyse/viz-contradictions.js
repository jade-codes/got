// analyse/viz-contradictions.js — contradiction and redundancy cards

import { esc } from '../shared/utils.js';

export function renderContradictions(turn) {
  const el = document.getElementById('tab-contradictions');
  if (turn.all_contradictions.length === 0) {
    el.innerHTML = '<div class="empty-state"><p>No contradictions at this point in the conversation</p></div>';
    return;
  }
  const newKeys = new Set(turn.new_contradictions.map(c => c.term_a + '|' + c.term_b));
  const activeKeys = new Set((turn.turn_contradictions || []).map(c => c.term_a + '|' + c.term_b));
  const sorted = [...turn.all_contradictions].sort((a, b) => b.severity - a.severity);
  el.innerHTML = sorted.map(c => {
    const sev = c.severity >= 0.5 ? 'severe' : c.severity >= 0.2 ? 'moderate' : 'mild';
    const key = c.term_a + '|' + c.term_b;
    const isNew = newKeys.has(key);
    const isActive = activeKeys.has(key);
    return '<div class="finding-card ' + sev + (isNew ? ' is-new' : '') + (isActive ? ' is-active' : '') + '">' +
      '<div class="finding-header">' +
      '<span class="finding-terms">\u201C' + esc(c.term_a) + '\u201D \u2194 \u201C' + esc(c.term_b) + '\u201D</span>' +
      '<span>' +
      '<span class="finding-badge ' + sev + '">' + sev + '</span>' +
      (isNew ? '<span class="finding-badge new-badge">new</span>' : '') +
      (isActive ? '<span class="finding-badge active-badge">active</span>' : '') +
      '</span></div>' +
      '<div class="finding-metrics">' +
      'cosine: ' + c.causal_cosine.toFixed(3) + ' &nbsp;|&nbsp; ' +
      'angle: ' + c.angle_degrees.toFixed(1) + '&deg; &nbsp;|&nbsp; ' +
      'severity: ' + c.severity.toFixed(2) + '</div></div>';
  }).join('');
}

export function renderRedundancies(turn) {
  const el = document.getElementById('tab-redundancies');
  if (turn.all_redundancies.length === 0) {
    el.innerHTML = '<div class="empty-state"><p>No redundancies at this point</p></div>';
    return;
  }
  el.innerHTML = turn.all_redundancies.map(r =>
    '<div class="finding-card redundant">' +
    '<div class="finding-header">' +
    '<span class="finding-terms">\u201C' + esc(r.term_a) + '\u201D \u2248 \u201C' + esc(r.term_b) + '\u201D</span>' +
    '<span class="finding-badge redundant">redundant</span></div>' +
    '<div class="finding-metrics">' +
    'similarity: ' + r.similarity.toFixed(3) + ' &nbsp;|&nbsp; ' +
    'cosine: ' + r.causal_cosine.toFixed(3) + '</div></div>'
  ).join('');
}
