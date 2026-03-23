// analyse/viz-heatmap.js — heatmap visualization

import { esc, getTermList } from '../shared/utils.js';

export function renderHeatmap(turn) {
  const container = document.getElementById('heatmapViz');
  const tooltip = document.getElementById('tooltip');
  container.innerHTML = '';
  const terms = getTermList(turn);
  const n = terms.length;
  if (n < 2) { container.innerHTML = '<div class="empty-state"><p>Need 2+ resolved terms</p></div>'; return; }

  const cellSize = Math.min(44, Math.max(18, 400 / n));
  const margin = { top: 90, right: 20, bottom: 20, left: 90 };
  const gridSize = n * cellSize;
  const width = gridSize + margin.left + margin.right;
  const height = gridSize + margin.top + margin.bottom;

  const svg = d3.select(container).append('svg').attr('width', width).attr('height', height);
  const g = svg.append('g').attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  const cosineMap = {};
  turn.pairwise.forEach(p => {
    cosineMap[p.term_a + '|' + p.term_b] = p.causal_cosine;
    cosineMap[p.term_b + '|' + p.term_a] = p.causal_cosine;
  });

  const colourScale = d3.scaleDiverging().domain([-1, 0, 1]).interpolator(d3.interpolateRdYlGn);

  const cells = [];
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i === j) { cells.push({ i, j, val: 1.0 }); continue; }
      cells.push({ i, j, val: cosineMap[terms[i] + '|' + terms[j]] || 0 });
    }
  }

  g.selectAll('rect').data(cells).join('rect')
    .attr('x', d => d.j * cellSize).attr('y', d => d.i * cellSize)
    .attr('width', cellSize - 1).attr('height', cellSize - 1).attr('rx', 2)
    .attr('fill', d => colourScale(d.val))
    .on('mouseover', (event, d) => {
      tooltip.style.opacity = 1;
      tooltip.innerHTML = '<strong>' + esc(terms[d.i]) + '</strong> \u2194 <strong>' + esc(terms[d.j]) + '</strong><br>cosine: ' + d.val.toFixed(3);
    })
    .on('mousemove', event => {
      tooltip.style.left = (event.pageX + 12) + 'px';
      tooltip.style.top = (event.pageY - 10) + 'px';
    })
    .on('mouseout', () => { tooltip.style.opacity = 0; });

  g.selectAll('.row-label').data(terms).join('text')
    .attr('class', 'row-label').attr('x', -6)
    .attr('y', (d, i) => i * cellSize + cellSize / 2)
    .attr('dy', '0.35em').attr('text-anchor', 'end')
    .attr('fill', '#c9d1d9').attr('font-size', Math.min(11, cellSize - 4) + 'px')
    .text(d => d);

  g.selectAll('.col-label').data(terms).join('text')
    .attr('class', 'col-label')
    .attr('transform', (d, i) =>
      'translate(' + (i * cellSize + cellSize / 2) + ', -6) rotate(-45)')
    .attr('text-anchor', 'start')
    .attr('fill', '#c9d1d9').attr('font-size', Math.min(11, cellSize - 4) + 'px')
    .text(d => d);

  // Legend
  const legendWidth = 180;
  const legendG = svg.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + (margin.top + gridSize + 10) + ')');
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', 'heatGrad2');
  grad.append('stop').attr('offset', '0%').attr('stop-color', colourScale(-1));
  grad.append('stop').attr('offset', '50%').attr('stop-color', colourScale(0));
  grad.append('stop').attr('offset', '100%').attr('stop-color', colourScale(1));
  legendG.append('rect').attr('width', legendWidth).attr('height', 10).attr('rx', 3).attr('fill', 'url(#heatGrad2)');
  legendG.append('text').attr('y', 22).attr('fill', '#8b949e').attr('font-size', '9px').text('-1 (opposed)');
  legendG.append('text').attr('x', legendWidth).attr('y', 22).attr('fill', '#8b949e').attr('font-size', '9px').attr('text-anchor', 'end').text('+1 (aligned)');
}
