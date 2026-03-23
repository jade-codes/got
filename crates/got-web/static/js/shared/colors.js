// shared/colors.js — color mapping functions

export function scoreColour(s) {
  if (s >= 0.9) return '#3fb950';
  if (s >= 0.7) return '#d29922';
  if (s >= 0.4) return '#f0883e';
  return '#f85149';
}

export function trustColour(t) {
  if (t >= 0.8) return '#3fb950';
  if (t >= 0.5) return '#d29922';
  if (t >= 0.2) return '#f0883e';
  return '#f85149';
}

export function arcColour(rel) {
  if (rel === 'Opposed') return '#f85149';
  if (rel === 'Aligned') return '#3fb950';
  return '#484f58';
}
