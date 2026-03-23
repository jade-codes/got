// analyse/viz-chord.js — chord diagram visualization

import { getTermList, findPair } from '../shared/utils.js';

export function renderChordDiagram(turn) {
  const container = document.getElementById('chordViz');
  container.innerHTML = '';
  const terms = getTermList(turn);
  const n = terms.length;
  if (n < 2) { container.innerHTML = '<div class="empty-state"><p>Need 2+ resolved terms</p></div>'; return; }

  const size = Math.min(480, window.innerWidth - 540);
  const outerRadius = size / 2 - 40;
  const innerRadius = outerRadius - 18;

  const matrix = [];
  for (let i = 0; i < n; i++) {
    matrix[i] = [];
    for (let j = 0; j < n; j++) {
      if (i === j) { matrix[i][j] = 0; continue; }
      const p = findPair(turn.pairwise, terms[i], terms[j]);
      matrix[i][j] = p ? Math.abs(p.causal_cosine) : 0;
    }
  }

  const svg = d3.select(container).append('svg')
    .attr('width', size).attr('height', size)
    .append('g').attr('transform', 'translate(' + size/2 + ',' + size/2 + ')');

  const chord = d3.chord().padAngle(0.05).sortSubgroups(d3.descending);
  const chords = chord(matrix);
  const arc = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
  const ribbon = d3.ribbon().radius(innerRadius);
  const colour = d3.scaleOrdinal(d3.schemeTableau10);

  svg.append('g').selectAll('path').data(chords.groups).join('path')
    .attr('d', arc).attr('fill', d => colour(d.index)).attr('stroke', '#0d1117');

  svg.append('g').selectAll('text').data(chords.groups).join('text')
    .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
    .attr('dy', '0.35em')
    .attr('transform', d =>
      'rotate(' + (d.angle * 180 / Math.PI - 90) + ')' +
      'translate(' + (outerRadius + 8) + ')' +
      (d.angle > Math.PI ? 'rotate(180)' : ''))
    .attr('text-anchor', d => d.angle > Math.PI ? 'end' : null)
    .attr('fill', '#c9d1d9').attr('font-size', '11px')
    .text(d => terms[d.index]);

  svg.append('g').attr('fill-opacity', 0.6).selectAll('path')
    .data(chords).join('path').attr('d', ribbon)
    .attr('fill', d => {
      const p = findPair(turn.pairwise, terms[d.source.index], terms[d.target.index]);
      if (!p) return '#484f58';
      if (p.relation === 'Opposed') return '#f85149';
      if (p.relation === 'Aligned') return '#3fb950';
      return '#484f58';
    })
    .attr('stroke', '#0d1117').attr('stroke-width', 0.5)
    .append('title').text(d => {
      const p = findPair(turn.pairwise, terms[d.source.index], terms[d.target.index]);
      return p ? p.term_a + ' \u2194 ' + p.term_b + '\ncos: ' + p.causal_cosine.toFixed(3) + ' (' + p.relation + ')' : '';
    });
}
