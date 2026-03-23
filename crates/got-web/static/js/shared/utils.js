// shared/utils.js — common helper functions

export function esc(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

export function getTermList(turn) {
  const s = new Set();
  turn.pairwise.forEach(p => { s.add(p.term_a); s.add(p.term_b); });
  return [...s].sort();
}

export function findPair(pairwise, a, b) {
  return pairwise.find(p =>
    (p.term_a === a && p.term_b === b) || (p.term_a === b && p.term_b === a));
}
