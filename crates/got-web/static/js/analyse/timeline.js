// analyse/timeline.js — coherence timeline chart

import { scoreColour } from '../shared/colors.js';

let timelineScales = null;

export function renderTimeline(analysis, selectTurnFn) {
  const timelineChart = document.getElementById('timelineChart');
  if (!analysis) return;
  timelineChart.innerHTML = '';

  const turns = analysis.turns;
  const w = timelineChart.clientWidth;
  const h = timelineChart.clientHeight;
  const margin = { top: 6, right: 12, bottom: 18, left: 6 };
  const innerW = w - margin.left - margin.right;
  const innerH = h - margin.top - margin.bottom;

  const svg = d3.select(timelineChart).append('svg')
    .attr('width', w).attr('height', h);

  const g = svg.append('g')
    .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

  const x = d3.scaleLinear().domain([0, turns.length - 1]).range([0, innerW]);
  const y = d3.scaleLinear().domain([0, 1]).range([innerH, 0]);

  // Area fill
  const area = d3.area()
    .x((d, i) => x(i))
    .y0(innerH)
    .y1(d => y(d.coherence_score))
    .curve(d3.curveMonotoneX);

  const defs = svg.append('defs');
  const grad = defs.append('linearGradient')
    .attr('id', 'areaGrad').attr('x1', '0').attr('x2', '0').attr('y1', '0').attr('y2', '1');
  grad.append('stop').attr('offset', '0%').attr('stop-color', '#3fb950').attr('stop-opacity', 0.3);
  grad.append('stop').attr('offset', '100%').attr('stop-color', '#f85149').attr('stop-opacity', 0.05);

  g.append('path').datum(turns).attr('d', area).attr('fill', 'url(#areaGrad)');

  // Coherence line
  const line = d3.line()
    .x((d, i) => x(i))
    .y(d => y(d.coherence_score))
    .curve(d3.curveMonotoneX);

  g.append('path').datum(turns)
    .attr('d', line).attr('fill', 'none')
    .attr('stroke', '#58a6ff').attr('stroke-width', 2);

  // Trust line
  const trustLine = d3.line()
    .x((d, i) => x(i))
    .y(d => y(d.trust_score))
    .curve(d3.curveMonotoneX);

  g.append('path').datum(turns)
    .attr('d', trustLine).attr('fill', 'none')
    .attr('stroke', '#f0883e').attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '6,3');

  // Message coherence line
  const msgCohLine = d3.line()
    .x((d, i) => x(i))
    .y(d => y(d.message_coherence))
    .curve(d3.curveMonotoneX);

  g.append('path').datum(turns)
    .attr('d', msgCohLine).attr('fill', 'none')
    .attr('stroke', '#3fb950').attr('stroke-width', 1.5)
    .attr('stroke-dasharray', '3,3');

  // Dots
  g.selectAll('circle.dot')
    .data(turns)
    .join('circle')
    .attr('class', 'dot')
    .attr('cx', (d, i) => x(i))
    .attr('cy', d => y(d.coherence_score))
    .attr('r', 4)
    .attr('fill', d => scoreColour(d.coherence_score))
    .attr('stroke', '#0d1117').attr('stroke-width', 1.5)
    .style('cursor', 'pointer')
    .on('click', (event, d) => selectTurnFn(d.turn));

  // Highlight indicator
  g.append('circle')
    .attr('id', 'timelineHighlight')
    .attr('r', 7).attr('fill', 'none')
    .attr('stroke', '#f0f6fc').attr('stroke-width', 2)
    .attr('opacity', 0);

  // Turn labels
  g.append('g')
    .attr('transform', 'translate(0,' + innerH + ')')
    .selectAll('text')
    .data(turns)
    .join('text')
    .attr('x', (d, i) => x(i))
    .attr('y', 14)
    .attr('text-anchor', 'middle')
    .attr('fill', '#484f58')
    .attr('font-size', '9px')
    .text((d, i) => i);

  timelineScales = { x, y };
}

export function updateTimelineHighlight(idx, analysis) {
  if (!analysis || !timelineScales) return;
  const { x, y } = timelineScales;
  const turn = analysis.turns[idx];
  const highlight = d3.select('#timelineHighlight');
  highlight
    .attr('cx', x(idx))
    .attr('cy', y(turn.coherence_score))
    .attr('opacity', 1)
    .attr('stroke', scoreColour(turn.coherence_score));
}
