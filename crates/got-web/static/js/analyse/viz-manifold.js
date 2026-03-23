// analyse/viz-manifold.js — manifold terrain visualization
//
// Renders a 2D density terrain: terms positioned by MDS on causal distances,
// density interpolated across the surface as a colormap with contour lines.
// Dense = peaks (bright), sparse = valleys (dark).

import { esc, getTermList, findPair } from '../shared/utils.js';

export function renderManifoldTerrain(container, turn, snapshot) {
  container.innerHTML = '';
  const hasDensity = snapshot && snapshot.term_densities && Object.keys(snapshot.term_densities).length >= 2;
  const hasWeights = snapshot && snapshot.term_weights && Object.keys(snapshot.term_weights).length >= 2;
  if (!hasDensity && !hasWeights) {
    container.innerHTML = '<div class="empty-state"><p>Need manifold data — click Attest Manifold</p></div>';
    return;
  }

  const terms = getTermList(turn);
  const n = terms.length;
  if (n < 2) {
    container.innerHTML = '<div class="empty-state"><p>Need 2+ terms</p></div>';
    return;
  }

  const size = Math.min(500, container.clientWidth || 460);
  const margin = 50;
  const innerSize = size - 2 * margin;

  // --- MDS: project terms to 2D from pairwise causal distances ---
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

  // Double-centering for MDS
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

  // Power iteration for top 2 eigenvectors
  const coords = mds2(B, n);

  // Normalize to [0, 1]
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < n; i++) {
    if (coords[i][0] < minX) minX = coords[i][0];
    if (coords[i][0] > maxX) maxX = coords[i][0];
    if (coords[i][1] < minY) minY = coords[i][1];
    if (coords[i][1] > maxY) maxY = coords[i][1];
  }
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;
  // Per-term: activation weight (EWMA, primary signal) + manifold density (secondary)
  const weights = snapshot.term_weights || {};
  const densities = snapshot.term_densities || {};

  const points = terms.map((term, i) => ({
    term,
    x: margin + ((coords[i][0] - minX) / rangeX) * innerSize,
    y: margin + ((coords[i][1] - minY) / rangeY) * innerSize,
    weight: weights[term] !== undefined ? weights[term] : null,
    density: densities[term] !== undefined ? densities[term] : null,
  }));

  // Normalize weights and densities to [0, 1]
  const weightVals = points.filter(p => p.weight !== null).map(p => p.weight);
  const densityVals = points.filter(p => p.density !== null).map(p => p.density);

  const minW = weightVals.length > 0 ? Math.min(...weightVals) : 0;
  const maxW = weightVals.length > 0 ? Math.max(...weightVals) : 1;
  const rangeW = maxW - minW || 1;

  const minD = densityVals.length > 0 ? Math.min(...densityVals) : 0;
  const maxD = densityVals.length > 0 ? Math.max(...densityVals) : 1;
  const rangeD = maxD - minD || 1;

  // Height = activation weight, used for terrain color
  // Each point contributes a gaussian splat weighted by its activation strength
  const res = 80;
  const grid = new Float64Array(res * res);
  const sigma = innerSize / 6;

  for (let gy = 0; gy < res; gy++) {
    for (let gx = 0; gx < res; gx++) {
      const px = margin + (gx / (res - 1)) * innerSize;
      const py = margin + (gy / (res - 1)) * innerSize;

      let kernelSum = 0;
      let valSum = 0;
      for (const pt of points) {
        // Use weight if available, fall back to density
        const h = pt.weight !== null ? (pt.weight - minW) / rangeW
          : pt.density !== null ? (pt.density - minD) / rangeD
          : 0;
        const dx = px - pt.x;
        const dy = py - pt.y;
        const k = Math.exp(-(dx * dx + dy * dy) / (2 * sigma * sigma));
        kernelSum += k;
        valSum += k * h;
      }
      grid[gy * res + gx] = kernelSum > 1e-12 ? valSum / kernelSum : 0;
    }
  }

  // --- Render with canvas (terrain) + SVG overlay (contours, labels) ---
  const wrapper = document.createElement('div');
  wrapper.style.position = 'relative';
  wrapper.style.width = size + 'px';
  wrapper.style.height = size + 'px';

  // Canvas terrain
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  canvas.style.borderRadius = '8px';
  const ctx = canvas.getContext('2d');

  // Color ramp: deep purple (sparse) -> teal -> bright green (dense)
  function terrainColor(t) {
    // t in [0, 1]: 0 = sparse, 1 = dense
    const r = Math.round(20 + (1 - t) * 50 + t * 40);
    const g = Math.round(10 + t * 200);
    const b = Math.round(40 + (1 - t) * 80 - t * 20);
    return `rgb(${r},${g},${Math.max(0, b)})`;
  }

  const cellW = size / res;
  const cellH = size / res;
  for (let gy = 0; gy < res; gy++) {
    for (let gx = 0; gx < res; gx++) {
      const v = grid[gy * res + gx];
      ctx.fillStyle = terrainColor(v);
      ctx.fillRect(gx * cellW, gy * cellH, cellW + 1, cellH + 1);
    }
  }

  wrapper.appendChild(canvas);

  // SVG overlay for contours and term labels
  const svg = d3.select(wrapper).append('svg')
    .attr('width', size).attr('height', size)
    .style('position', 'absolute').style('top', '0').style('left', '0');

  // Contour lines
  const thresholds = [0.2, 0.4, 0.6, 0.8];
  const contourGen = d3.contours().size([res, res]).thresholds(thresholds);
  const contours = contourGen(grid);

  const xScale = d3.scaleLinear().domain([0, res]).range([0, size]);
  const yScale = d3.scaleLinear().domain([0, res]).range([0, size]);

  svg.selectAll('path.contour').data(contours).join('path')
    .attr('class', 'contour')
    .attr('d', d3.geoPath().projection(d3.geoTransform({
      point: function(x, y) { this.stream.point(xScale(x), yScale(y)); }
    })))
    .attr('fill', 'none')
    .attr('stroke', 'rgba(255,255,255,0.25)')
    .attr('stroke-width', 1);

  // Term dots and labels
  const dotG = svg.append('g');

  points.forEach(pt => {
    const wn = pt.weight !== null ? (pt.weight - minW) / rangeW : 0.5;
    const dn = pt.density !== null ? (pt.density - minD) / rangeD : 0.5;
    const r = 4 + wn * 8; // size = activation weight

    // Glow proportional to weight
    if (wn > 0.2) {
      dotG.append('circle')
        .attr('cx', pt.x).attr('cy', pt.y).attr('r', r + 6 + wn * 4)
        .attr('fill', 'none').attr('stroke', `rgba(255,255,255,${0.08 + wn * 0.15})`)
        .attr('stroke-width', 1.5 + wn * 2);
    }

    // Dot — bright if on-manifold (high density), dim if off-manifold
    const brightness = 0.4 + dn * 0.6;
    const c = Math.round(brightness * 240);
    dotG.append('circle')
      .attr('cx', pt.x).attr('cy', pt.y).attr('r', r)
      .attr('fill', `rgb(${c},${c},${Math.round(c * 0.95)})`)
      .attr('stroke', '#0d1117').attr('stroke-width', 1.5)
      .append('title').text(pt.term +
        (pt.weight !== null ? '\nweight: ' + pt.weight.toFixed(3) : '') +
        (pt.density !== null ? '\ndensity: ' + pt.density.toFixed(2) : ''));

    // Label — larger for higher-weight terms
    const fontSize = 10 + wn * 3;
    dotG.append('text')
      .attr('x', pt.x).attr('y', pt.y - r - 5)
      .attr('text-anchor', 'middle')
      .attr('fill', '#f0f6fc').attr('font-size', fontSize + 'px')
      .attr('font-weight', wn > 0.5 ? '700' : '500')
      .style('text-shadow', '0 0 4px #0d1117, 0 0 8px #0d1117')
      .text(pt.term);
  });

  // Legend
  const legendG = svg.append('g').attr('transform', `translate(${margin}, ${size - 18})`);
  const legendW = innerSize;
  const gradId = 'terrainGrad';
  const defs = svg.append('defs');
  const grad = defs.append('linearGradient').attr('id', gradId);
  grad.append('stop').attr('offset', '0%').attr('stop-color', terrainColor(0));
  grad.append('stop').attr('offset', '50%').attr('stop-color', terrainColor(0.5));
  grad.append('stop').attr('offset', '100%').attr('stop-color', terrainColor(1));
  legendG.append('rect').attr('width', legendW).attr('height', 8)
    .attr('rx', 3).attr('fill', `url(#${gradId})`);
  legendG.append('text').attr('y', -3)
    .attr('fill', '#8b949e').attr('font-size', '9px').text('low activation');
  legendG.append('text').attr('x', legendW).attr('y', -3)
    .attr('text-anchor', 'end')
    .attr('fill', '#8b949e').attr('font-size', '9px').text('high activation');
  legendG.append('text').attr('x', legendW / 2).attr('y', -3)
    .attr('text-anchor', 'middle')
    .attr('fill', '#6e7681').attr('font-size', '8px').text('dot size = weight | brightness = density');

  container.appendChild(wrapper);
}

// --- MDS helper: extract top 2 eigenvectors via power iteration ---
function mds2(B, n) {
  const coords = [];
  const mat = B.map(row => Float64Array.from(row));
  for (let dim = 0; dim < 2; dim++) {
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
      if (!coords[i]) coords[i] = [0, 0];
      coords[i][dim] = v[i] * scale;
    }
    // Deflate
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        mat[i][j] -= eigenvalue * v[i] * v[j];
      }
    }
  }
  return coords;
}
