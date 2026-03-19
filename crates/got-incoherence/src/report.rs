// ---------------------------------------------------------------------------
// Reporting: structured output from coherence analysis.
//
// Converts a CoherenceAnalysis into human-readable and machine-readable
// formats.  Designed for two consumers:
//   1. CLI users → plaintext summary with colour-coded severity
//   2. Downstream tools → JSON serialisation (via serde on CoherenceAnalysis)
//
// The report is deterministic: same analysis → same output string.
// ---------------------------------------------------------------------------

use crate::coherence::{
    CoherenceAnalysis, Contradiction, ConversationAnalysis, Redundancy, RelationType,
};

// ---------------------------------------------------------------------------
// Plaintext report
// ---------------------------------------------------------------------------

/// Render a coherence analysis as a human-readable plaintext report.
pub fn render_text(analysis: &CoherenceAnalysis) -> String {
    let mut out = String::new();

    // Header
    out.push_str("=== Value System Coherence Report ===\n\n");
    out.push_str(&format!(
        "Terms analysed: {}  |  Unresolved: {}\n",
        analysis.num_terms, analysis.num_unresolved
    ));
    out.push_str(&format!(
        "Coherence score: {:.2} / 1.00",
        analysis.coherence_score
    ));
    out.push_str(&format!("  [{}]\n\n", score_label(analysis.coherence_score)));

    // Contradictions
    if analysis.contradictions.is_empty() {
        out.push_str("Contradictions: none detected\n\n");
    } else {
        out.push_str(&format!(
            "Contradictions: {} detected\n",
            analysis.contradictions.len()
        ));
        let mut sorted = analysis.contradictions.clone();
        sorted.sort_by(|a, b| b.severity.partial_cmp(&a.severity).unwrap_or(std::cmp::Ordering::Equal));
        for (i, c) in sorted.iter().enumerate() {
            out.push_str(&format_contradiction(i + 1, c));
        }
        out.push('\n');
    }

    // Redundancies
    if !analysis.redundancies.is_empty() {
        out.push_str(&format!(
            "Redundancies: {} detected\n",
            analysis.redundancies.len()
        ));
        for (i, r) in analysis.redundancies.iter().enumerate() {
            out.push_str(&format_redundancy(i + 1, r));
        }
        out.push('\n');
    }

    // Pairwise matrix summary
    out.push_str("Pairwise relations:\n");
    let opposed = analysis.pairwise.iter().filter(|r| r.relation == RelationType::Opposed).count();
    let aligned = analysis.pairwise.iter().filter(|r| r.relation == RelationType::Aligned).count();
    let independent = analysis.pairwise.iter().filter(|r| r.relation == RelationType::Independent).count();
    out.push_str(&format!(
        "  Opposed: {}  |  Aligned: {}  |  Independent: {}\n",
        opposed, aligned, independent
    ));

    out
}

/// Render the analysis as pretty-printed JSON.
pub fn render_json(analysis: &CoherenceAnalysis) -> Result<String, serde_json::Error> {
    serde_json::to_string_pretty(analysis)
}

// ---------------------------------------------------------------------------
// Conversation report
// ---------------------------------------------------------------------------

/// Render a conversation analysis as a human-readable transcript with
/// inline coherence annotations.
///
/// Shows each message with the values it introduces, flags new
/// contradictions as they emerge, and includes a coherence sparkline
/// showing how the score evolves turn by turn.
pub fn render_conversation(conv: &ConversationAnalysis) -> String {
    let mut out = String::new();

    out.push_str("=== Conversational Incoherence Report ===\n\n");

    if conv.turns.is_empty() {
        out.push_str("(no turns)\n");
        return out;
    }

    // Summary line
    let final_score = conv.final_score();
    let total_turns = conv.turns.len();
    let total_contradictions = conv.total_contradictions();
    out.push_str(&format!(
        "Turns: {}  |  Final score: {:.2} [{}]  |  Contradictions: {}\n",
        total_turns,
        final_score,
        score_label(final_score),
        total_contradictions,
    ));

    // ASCII sparkline of coherence over time
    out.push_str("Coherence: ");
    let scores = conv.score_series();
    out.push_str(&ascii_sparkline(&scores));
    out.push('\n');

    // First contradiction turn
    if let Some(turn) = conv.first_contradiction_turn() {
        out.push_str(&format!("First contradiction at turn {turn}\n"));
    }
    out.push('\n');

    // Per-turn transcript
    out.push_str("--- Transcript ---\n\n");

    for turn in &conv.turns {
        // Speaker and turn number
        out.push_str(&format!(
            "[Turn {}] {} (score: {:.2})\n",
            turn.turn, turn.speaker, turn.analysis.coherence_score
        ));

        // Message text (indented)
        for line in turn.text.lines() {
            out.push_str(&format!("  {line}\n"));
        }

        // Values introduced
        if !turn.values_introduced.is_empty() {
            out.push_str(&format!(
                "  + values: {}\n",
                turn.values_introduced.join(", ")
            ));
        }

        // New contradictions at this turn
        for c in &turn.new_contradictions {
            let sev = if c.severity >= 0.8 {
                "SEVERE"
            } else if c.severity >= 0.5 {
                "MODERATE"
            } else {
                "MILD"
            };
            out.push_str(&format!(
                "  !! [{sev}] \"{a}\" <-> \"{b}\" (cos: {cos:.3}, severity: {s:.2})\n",
                a = c.term_a,
                b = c.term_b,
                cos = c.causal_cosine,
                s = c.severity,
            ));
        }

        out.push('\n');
    }

    // Final full contradiction list
    if let Some(last) = conv.turns.last() {
        if !last.analysis.contradictions.is_empty() {
            out.push_str("--- All Contradictions (final state) ---\n\n");
            let mut sorted = last.analysis.contradictions.clone();
            sorted.sort_by(|a, b| {
                b.severity
                    .partial_cmp(&a.severity)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            for (i, c) in sorted.iter().enumerate() {
                out.push_str(&format_contradiction(i + 1, c));
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn score_label(score: f32) -> &'static str {
    if score >= 0.9 {
        "COHERENT"
    } else if score >= 0.7 {
        "MOSTLY COHERENT"
    } else if score >= 0.4 {
        "INCOHERENT"
    } else {
        "SEVERELY INCOHERENT"
    }
}

/// Render a series of [0, 1] values as an ASCII sparkline.
///
/// Uses Unicode block characters ▁▂▃▄▅▆▇█ to show the trend.
fn ascii_sparkline(values: &[f32]) -> String {
    const BLOCKS: &[char] = &['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    values
        .iter()
        .map(|&v| {
            let idx = ((v.clamp(0.0, 1.0)) * (BLOCKS.len() - 1) as f32).round() as usize;
            BLOCKS[idx]
        })
        .collect()
}

fn format_contradiction(idx: usize, c: &Contradiction) -> String {
    let severity_label = if c.severity >= 0.8 {
        "SEVERE"
    } else if c.severity >= 0.5 {
        "MODERATE"
    } else {
        "MILD"
    };

    format!(
        "  {idx}. [{severity_label}] \"{term_a}\" <-> \"{term_b}\"\n\
         \x20    cosine: {cos:.3}  |  angle: {angle:.1}°  |  severity: {sev:.2}\n",
        term_a = c.term_a,
        term_b = c.term_b,
        cos = c.causal_cosine,
        angle = c.angle_degrees,
        sev = c.severity,
    )
}

fn format_redundancy(idx: usize, r: &Redundancy) -> String {
    format!(
        "  {idx}. \"{term_a}\" ≈ \"{term_b}\"  (similarity: {sim:.3})\n",
        term_a = r.term_a,
        term_b = r.term_b,
        sim = r.similarity,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coherence::{CoherenceAnalysis, Contradiction, PairwiseRelation, Redundancy, RelationType};

    fn sample_analysis() -> CoherenceAnalysis {
        CoherenceAnalysis {
            pairwise: vec![
                PairwiseRelation {
                    term_a: "innovation".into(),
                    term_b: "risk-aversion".into(),
                    causal_cosine: -0.85,
                    causal_distance: 3.5,
                    relation: RelationType::Opposed,
                },
                PairwiseRelation {
                    term_a: "innovation".into(),
                    term_b: "creativity".into(),
                    causal_cosine: 0.92,
                    causal_distance: 0.5,
                    relation: RelationType::Aligned,
                },
                PairwiseRelation {
                    term_a: "risk-aversion".into(),
                    term_b: "creativity".into(),
                    causal_cosine: -0.1,
                    causal_distance: 2.0,
                    relation: RelationType::Independent,
                },
            ],
            contradictions: vec![Contradiction {
                term_a: "innovation".into(),
                term_b: "risk-aversion".into(),
                severity: 0.7,
                causal_cosine: -0.85,
                angle_degrees: 148.2,
            }],
            redundancies: vec![Redundancy {
                term_a: "innovation".into(),
                term_b: "creativity".into(),
                similarity: 0.92,
                causal_cosine: 0.92,
            }],
            coherence_score: 0.77,
            num_terms: 3,
            num_unresolved: 0,
        }
    }

    #[test]
    fn text_report_contains_score() {
        let report = render_text(&sample_analysis());
        assert!(report.contains("0.77"), "report should contain score: {report}");
        assert!(report.contains("MOSTLY COHERENT"), "report should classify score");
    }

    #[test]
    fn text_report_lists_contradictions() {
        let report = render_text(&sample_analysis());
        assert!(report.contains("innovation"), "should mention innovation");
        assert!(report.contains("risk-aversion"), "should mention risk-aversion");
        assert!(report.contains("MODERATE"), "should classify severity");
    }

    #[test]
    fn text_report_lists_redundancies() {
        let report = render_text(&sample_analysis());
        assert!(report.contains("creativity"), "should mention creativity in redundancies");
    }

    #[test]
    fn json_report_roundtrips() {
        let analysis = sample_analysis();
        let json = render_json(&analysis).unwrap();
        let parsed: CoherenceAnalysis = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.coherence_score, analysis.coherence_score);
        assert_eq!(parsed.contradictions.len(), 1);
    }

    #[test]
    fn score_labels_correct() {
        assert_eq!(score_label(0.95), "COHERENT");
        assert_eq!(score_label(0.75), "MOSTLY COHERENT");
        assert_eq!(score_label(0.5), "INCOHERENT");
        assert_eq!(score_label(0.2), "SEVERELY INCOHERENT");
    }

    #[test]
    fn no_contradictions_report() {
        let analysis = CoherenceAnalysis {
            pairwise: vec![],
            contradictions: vec![],
            redundancies: vec![],
            coherence_score: 1.0,
            num_terms: 3,
            num_unresolved: 0,
        };
        let report = render_text(&analysis);
        assert!(report.contains("none detected"));
    }
}
