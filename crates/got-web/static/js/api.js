// api.js — all fetch() wrappers for the GOT API

export async function fetchDemoConversation() {
  const res = await fetch('/api/demo-conversation');
  return res.json();
}

export async function analyseConversation(messages, options = {}) {
  const body = { messages, ...options };
  const res = await fetch('/api/conversation/analyse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function chatWithModel(provider, apiKey, model, messages, baseUrl) {
  const body = { provider, api_key: apiKey, model, messages };
  if (baseUrl) body.base_url = baseUrl;
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function embedText(text) {
  const res = await fetch('/api/embed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  return res.json();
}

export async function createProxySession(targetModelId, sessionId) {
  const body = { target_model_id: targetModelId };
  if (sessionId) body.session_id = sessionId;
  const res = await fetch('/api/proxy/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function proxyObserve(sessionId, embedding, speaker = 'assistant') {
  const res = await fetch(`/api/proxy/session/${sessionId}/observe`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ embedding, speaker }),
  });
  return res.json();
}

export async function proxyStatus(sessionId) {
  const res = await fetch(`/api/proxy/session/${sessionId}/status`);
  return res.json();
}

export async function proxyHistory(sessionId) {
  const res = await fetch(`/api/proxy/session/${sessionId}/history`);
  return res.json();
}

export async function proxyManifold(sessionId) {
  const res = await fetch(`/api/proxy/session/${sessionId}/manifold`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: '{}',
  });
  return res.json();
}

export async function fetchCoherence(ordering, embeddings = [], sharpness = 1.0) {
  const res = await fetch('/api/coherence', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ordering, embeddings, sharpness }),
  });
  return res.json();
}

export async function fetchCollapse(probeTerms = []) {
  const res = await fetch('/api/collapse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ probe_terms: probeTerms }),
  });
  return res.json();
}

export async function fetchCompare(comparisonGotuePath, probeTerms = []) {
  const res = await fetch('/api/compare', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ comparison_gotue_path: comparisonGotuePath, probe_terms: probeTerms }),
  });
  return res.json();
}

export async function proxySnapshot(sessionId, attestationType) {
  const body = {};
  if (attestationType) body.attestation_type = attestationType;
  const res = await fetch(`/api/proxy/session/${sessionId}/snapshot`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return res.json();
}
