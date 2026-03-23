// analyse/viz-sphere.js — 3D MDS sphere visualization

import { arcColour } from '../shared/colors.js';
import { getTermList, findPair } from '../shared/utils.js';

function mds3(B, n) {
  const coords = [];
  const mat = B.map(row => Float64Array.from(row));
  for (let dim = 0; dim < 3; dim++) {
    let v = new Float64Array(n);
    for (let i = 0; i < n; i++) v[i] = Math.random() - 0.5;
    let eigenvalue = 0;
    for (let iter = 0; iter < 200; iter++) {
      const w = new Float64Array(n);
      for (let i = 0; i < n; i++) {
        let s = 0;
        for (let j = 0; j < n; j++) s += mat[i][j] * v[j];
        w[i] = s;
      }
      let norm = 0;
      for (let i = 0; i < n; i++) norm += w[i] * w[i];
      norm = Math.sqrt(norm);
      if (norm < 1e-12) break;
      eigenvalue = norm;
      for (let i = 0; i < n; i++) v[i] = w[i] / norm;
    }
    const scale = eigenvalue > 0 ? Math.sqrt(eigenvalue) : 0;
    for (let i = 0; i < n; i++) {
      if (!coords[i]) coords[i] = [0, 0, 0];
      coords[i][dim] = v[i] * scale;
    }
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        mat[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }
  return coords;
}

export function renderSphere(turn, prevTurn) {
  const container = document.getElementById('sphereViz');
  container.innerHTML = '';
  const terms = getTermList(turn);
  const n = terms.length;
  if (n < 2) {
    container.innerHTML = '<div class="empty-state"><p>Need 2+ resolved terms</p></div>';
    return;
  }

  const size = Math.min(520, window.innerWidth - 540);
  const cx = size / 2, cy = size / 2;
  const sphereR = size / 2 - 60;

  const prevPairKeys = new Set();
  const prevActiveSet = new Set();
  if (prevTurn) {
    prevTurn.pairwise.forEach(p => {
      prevPairKeys.add(p.term_a + '|' + p.term_b);
      prevPairKeys.add(p.term_b + '|' + p.term_a);
    });
    (prevTurn.cumulative_values || []).forEach(v => prevActiveSet.add(v));
  }

  // Build squared-distance matrix
  const D2 = [];
  for (let i = 0; i < n; i++) D2[i] = new Float64Array(n);
  const termIdx = {};
  terms.forEach((t, i) => termIdx[t] = i);
  turn.pairwise.forEach(p => {
    const i = termIdx[p.term_a], j = termIdx[p.term_b];
    if (i !== undefined && j !== undefined) {
      const d2 = p.causal_distance * p.causal_distance;
      D2[i][j] = d2; D2[j][i] = d2;
    }
  });

  // Double-centering
  const rowMean = new Float64Array(n);
  const colMean = new Float64Array(n);
  let grandMean = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      rowMean[i] += D2[i][j]; colMean[j] += D2[i][j]; grandMean += D2[i][j];
    }
  }
  for (let i = 0; i < n; i++) { rowMean[i] /= n; colMean[i] /= n; }
  grandMean /= (n * n);

  const B = [];
  for (let i = 0; i < n; i++) {
    B[i] = new Float64Array(n);
    for (let j = 0; j < n; j++) {
      B[i][j] = -0.5 * (D2[i][j] - rowMean[i] - colMean[j] + grandMean);
    }
  }

  const coords3d = mds3(B, n);
  let maxR = 0;
  for (let i = 0; i < n; i++) {
    const r = Math.sqrt(coords3d[i][0]**2 + coords3d[i][1]**2 + coords3d[i][2]**2);
    if (r > maxR) maxR = r;
  }
  if (maxR > 0) {
    for (let i = 0; i < n; i++) {
      coords3d[i][0] /= maxR; coords3d[i][1] /= maxR; coords3d[i][2] /= maxR;
    }
  }

  const points = terms.map((term, i) => ({
    term, x: coords3d[i][0], y: coords3d[i][1], z: coords3d[i][2],
    r3d: Math.sqrt(coords3d[i][0]**2 + coords3d[i][1]**2 + coords3d[i][2]**2)
  }));

  const edges = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const p = findPair(turn.pairwise, terms[i], terms[j]);
      if (p) {
        const key = p.term_a + '|' + p.term_b;
        edges.push({ si: i, ti: j, relation: p.relation, cosine: p.causal_cosine,
          isNew: !prevPairKeys.has(key), isCarried: prevPairKeys.has(key) });
      }
    }
  }

  const active = new Set(turn.cumulative_values);
  const introduced = new Set(turn.values_introduced);

  let rotX = -0.3, rotY = 0.4;

  function project(pt) {
    let x = pt.x, y = pt.y, z = pt.z;
    let x1 = x * Math.cos(rotY) + z * Math.sin(rotY);
    let z1 = -x * Math.sin(rotY) + z * Math.cos(rotY);
    let y1 = y * Math.cos(rotX) - z1 * Math.sin(rotX);
    let z2 = y * Math.sin(rotX) + z1 * Math.cos(rotX);
    const fov = 3.5;
    const scale = fov / (fov + z2);
    return { px: cx + x1 * sphereR * scale, py: cy - y1 * sphereR * scale, z: z2, scale };
  }

  function drawWireRing(g, nx, ny, nz, colour, opacity) {
    const steps = 60;
    const ringPts = [];
    let ux, uy, uz;
    if (Math.abs(nx) < 0.9) { ux = 0; uy = nz; uz = -ny; }
    else { ux = -nz; uy = 0; uz = nx; }
    const uLen = Math.sqrt(ux*ux + uy*uy + uz*uz);
    ux /= uLen; uy /= uLen; uz /= uLen;
    const vx = ny*uz - nz*uy, vy = nz*ux - nx*uz, vz = nx*uy - ny*ux;
    for (let s = 0; s <= steps; s++) {
      const a = (s / steps) * Math.PI * 2;
      const proj = project({
        x: Math.cos(a) * ux + Math.sin(a) * vx,
        y: Math.cos(a) * uy + Math.sin(a) * vy,
        z: Math.cos(a) * uz + Math.sin(a) * vz
      });
      ringPts.push([proj.px, proj.py]);
    }
    g.append('path').attr('d', d3.line().curve(d3.curveBasisClosed)(ringPts))
      .attr('fill', 'none').attr('stroke', colour).attr('stroke-width', 0.5).attr('opacity', opacity);
  }

  const svg = d3.select(container).append('svg')
    .attr('width', size).attr('height', size).style('cursor', 'grab');

  const defs = svg.append('defs');
  const radGrad = defs.append('radialGradient').attr('id', 'sphereBg')
    .attr('cx', '40%').attr('cy', '35%');
  radGrad.append('stop').attr('offset', '0%').attr('stop-color', '#161b22');
  radGrad.append('stop').attr('offset', '100%').attr('stop-color', '#0d1117');

  svg.append('circle').attr('cx', cx).attr('cy', cy).attr('r', sphereR)
    .attr('fill', 'url(#sphereBg)').attr('stroke', '#30363d').attr('stroke-width', 1);

  const edgeGroup = svg.append('g');
  const dotGroup = svg.append('g');
  const labelGroup = svg.append('g');

  function render() {
    const projected = points.map(pt => ({ ...pt, ...project(pt) }));
    const sortedIdx = projected.map((_, i) => i).sort((a, b) => projected[a].z - projected[b].z);

    edgeGroup.selectAll('*').remove();
    dotGroup.selectAll('*').remove();
    labelGroup.selectAll('*').remove();

    drawWireRing(edgeGroup, 0, 1, 0, '#21262d', 0.3);
    drawWireRing(edgeGroup, 1, 0, 0, '#21262d', 0.2);
    drawWireRing(edgeGroup, 0, 0, 1, '#21262d', 0.2);

    edges.filter(e => e.isCarried).forEach(e => {
      const a = projected[e.si], b = projected[e.ti];
      edgeGroup.append('line')
        .attr('x1', a.px).attr('y1', a.py).attr('x2', b.px).attr('y2', b.py)
        .attr('stroke', arcColour(e.relation)).attr('stroke-width', 1)
        .attr('stroke-opacity', 0.2).attr('stroke-dasharray', '4,3')
        .append('title').text(points[e.si].term + ' \u2194 ' + points[e.ti].term +
          '  cos: ' + e.cosine.toFixed(3) + ' (' + e.relation + ') [previous]');
    });

    edges.filter(e => e.isNew).forEach(e => {
      const a = projected[e.si], b = projected[e.ti];
      edgeGroup.append('line')
        .attr('x1', a.px).attr('y1', a.py).attr('x2', b.px).attr('y2', b.py)
        .attr('stroke', arcColour(e.relation)).attr('stroke-width', 4).attr('stroke-opacity', 0.12);
      edgeGroup.append('line')
        .attr('x1', a.px).attr('y1', a.py).attr('x2', b.px).attr('y2', b.py)
        .attr('stroke', arcColour(e.relation)).attr('stroke-width', 2).attr('stroke-opacity', 0.8)
        .append('title').text(points[e.si].term + ' \u2194 ' + points[e.ti].term +
          '  cos: ' + e.cosine.toFixed(3) + ' (' + e.relation + ') [NEW]');
    });

    sortedIdx.forEach(i => {
      const pt = projected[i];
      const isActive = active.has(pt.term);
      const isNew = introduced.has(pt.term);
      const wasPrev = prevActiveSet.has(pt.term);
      const depthFade = 0.4 + 0.6 * ((pt.z + 1) / 2);

      let r, fill, stroke, sw;
      if (isNew) { r = 8 * pt.scale; fill = '#58a6ff'; stroke = '#79c0ff'; sw = 2.5; }
      else if (isActive && wasPrev) { r = 6 * pt.scale; fill = '#8b949e'; stroke = '#30363d'; sw = 1; }
      else if (isActive) { r = 7 * pt.scale; fill = '#f0f6fc'; stroke = '#30363d'; sw = 1; }
      else { r = 3 * pt.scale; fill = '#21262d'; stroke = '#30363d'; sw = 0.5; }

      if (isActive && pt.r3d > 0.05) {
        dotGroup.append('circle').attr('cx', pt.px).attr('cy', pt.py)
          .attr('r', r + 4 * pt.scale).attr('fill', 'none')
          .attr('stroke', isNew ? '#58a6ff' : '#30363d')
          .attr('stroke-width', 0.5).attr('opacity', 0.3 * depthFade);
      }

      dotGroup.append('circle').attr('cx', pt.px).attr('cy', pt.py).attr('r', r)
        .attr('fill', fill).attr('stroke', stroke).attr('stroke-width', sw)
        .attr('opacity', (isActive ? 1 : 0.3) * depthFade)
        .append('title').text(pt.term + ' (r=' + pt.r3d.toFixed(2) + ')' +
          (isNew ? ' [NEW]' : wasPrev ? ' [previous]' : ''));

      if (isActive || isNew) {
        labelGroup.append('text').attr('x', pt.px).attr('y', pt.py - r - 3)
          .attr('text-anchor', 'middle')
          .attr('fill', isNew ? '#58a6ff' : '#c9d1d9')
          .attr('font-size', (isNew ? 12 : 11) * pt.scale + 'px')
          .attr('font-weight', isNew ? '700' : '500')
          .attr('opacity', depthFade).text(pt.term);
      }
    });
  }

  render();

  // Drag to rotate
  let dragStartX, dragStartY, startRotX, startRotY;
  const drag = d3.drag()
    .on('start', function(event) {
      dragStartX = event.x; dragStartY = event.y;
      startRotX = rotX; startRotY = rotY;
      d3.select(this).style('cursor', 'grabbing');
    })
    .on('drag', function(event) {
      rotY = startRotY + (event.x - dragStartX) * 0.008;
      rotX = startRotX + (event.y - dragStartY) * 0.008;
      render();
    })
    .on('end', function() { d3.select(this).style('cursor', 'grab'); });

  svg.call(drag);

  // Legend
  const legend = d3.select(container).append('div')
    .style('display', 'flex').style('gap', '16px').style('padding', '10px 0')
    .style('justify-content', 'center').style('flex-wrap', 'wrap')
    .style('font-size', '11px').style('color', '#8b949e');

  [
    { label: 'New this turn', color: '#58a6ff', dash: false },
    { label: 'Previous', color: '#8b949e', dash: true },
    { label: 'Opposed', color: '#f85149', dash: false },
    { label: 'Aligned', color: '#3fb950', dash: false },
    { label: 'Near centre = mixed', color: '#484f58', dash: false },
  ].forEach(item => {
    const el = legend.append('span').style('display', 'inline-flex')
      .style('align-items', 'center').style('gap', '4px');
    const sw = el.append('svg').attr('width', 20).attr('height', 10);
    if (item.dash !== undefined) {
      sw.append('line').attr('x1', 0).attr('y1', 5).attr('x2', 20).attr('y2', 5)
        .attr('stroke', item.color).attr('stroke-width', 2)
        .attr('stroke-dasharray', item.dash ? '4,3' : 'none');
    } else {
      sw.append('circle').attr('cx', 10).attr('cy', 5).attr('r', 4).attr('fill', item.color);
    }
    el.append('span').text(item.label);
  });
}
