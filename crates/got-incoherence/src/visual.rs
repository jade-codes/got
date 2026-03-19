// ---------------------------------------------------------------------------
// Visual: SVG generation for coherence analysis results.
//
// Produces two visualizations:
//   1. Heatmap — pairwise causal cosine matrix, colour-coded.
//   2. Chord diagram — terms on a circle, arcs coloured by relationship.
//
// Output is standalone SVG (valid XML). No external dependencies.
// Can be embedded in HTML or opened directly in any browser.
// ---------------------------------------------------------------------------

use crate::coherence::{CoherenceAnalysis, ConversationAnalysis, RelationType};

// ---------------------------------------------------------------------------
// Colour mapping
// ---------------------------------------------------------------------------

/// Map a causal cosine ∈ [-1, 1] to an RGB hex colour.
///
///  -1.0 → #d32f2f (red, opposed)
///   0.0 → #f5f5f5 (light grey, independent)
///  +1.0 → #1565c0 (blue, aligned)
fn cosine_to_colour(cos: f32) -> String {
    let t = (cos.clamp(-1.0, 1.0) + 1.0) / 2.0; // map [-1,1] → [0,1]

    // Red channel: high at t=0 (opposed), low at t=1 (aligned)
    let r_lo: f32 = 21.0; // blue end
    let r_mid: f32 = 245.0; // neutral
    let r_hi: f32 = 211.0; // red end
    // Green channel
    let g_lo: f32 = 101.0;
    let g_mid: f32 = 245.0;
    let g_hi: f32 = 47.0;
    // Blue channel
    let b_lo: f32 = 192.0;
    let b_mid: f32 = 245.0;
    let b_hi: f32 = 47.0;

    let (r, g, b) = if t < 0.5 {
        // Opposed → neutral (red to grey)
        let s = t * 2.0;
        (
            lerp(r_hi, r_mid, s),
            lerp(g_hi, g_mid, s),
            lerp(b_hi, b_mid, s),
        )
    } else {
        // Neutral → aligned (grey to blue)
        let s = (t - 0.5) * 2.0;
        (
            lerp(r_mid, r_lo, s),
            lerp(g_mid, g_lo, s),
            lerp(b_mid, b_lo, s),
        )
    };

    format!(
        "#{:02x}{:02x}{:02x}",
        r.round() as u8,
        g.round() as u8,
        b.round() as u8
    )
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Colour for relationship arcs in the chord diagram.
fn relation_colour(relation: RelationType) -> &'static str {
    match relation {
        RelationType::Opposed => "#d32f2f",
        RelationType::Aligned => "#1565c0",
        RelationType::Independent => "#bdbdbd",
    }
}

/// Arc opacity based on strength of relationship.
fn relation_opacity(cos: f32) -> f32 {
    let strength = cos.abs();
    // Minimum 0.15 so independent lines are faintly visible.
    0.15 + 0.85 * strength
}

// ---------------------------------------------------------------------------
// Heatmap
// ---------------------------------------------------------------------------

/// Render a pairwise cosine heatmap as standalone SVG.
///
/// Returns a UTF-8 SVG string.  The matrix shows all N×N pairs with
/// self-similarity on the diagonal (always 1.0 / blue).
pub fn render_heatmap(analysis: &CoherenceAnalysis) -> String {
    let n = analysis.num_terms;
    if n == 0 {
        return empty_svg("No terms to visualise");
    }

    // Collect unique term names in order they first appear.
    let terms = ordered_terms(analysis);
    let n = terms.len();

    // Build cosine lookup: (term_a, term_b) → causal_cosine
    let lookup = cosine_lookup(analysis);

    let cell = 60u32; // pixels per cell
    let label_w = 120u32; // left label width
    let label_h = 120u32; // top label height (rotated)
    let legend_w = 80u32;
    let total_w = label_w + cell * n as u32 + legend_w + 20;
    let total_h = label_h + cell * n as u32 + 40;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" font-family="system-ui, sans-serif" font-size="12">"#
    ));
    svg.push_str(&format!(
        r#"<rect width="{total_w}" height="{total_h}" fill="white"/>"#
    ));

    // Title
    svg.push_str(&format!(
        r#"<text x="{}" y="18" text-anchor="middle" font-size="14" font-weight="bold">Value Coherence Heatmap (score: {:.2})</text>"#,
        total_w / 2,
        analysis.coherence_score
    ));

    // Column labels (rotated)
    for (j, term) in terms.iter().enumerate() {
        let x = label_w + j as u32 * cell + cell / 2;
        let y = label_h - 4;
        svg.push_str(&format!(
            r#"<text x="{x}" y="{y}" text-anchor="start" transform="rotate(-45,{x},{y})" font-size="11">{}</text>"#,
            xml_escape(term)
        ));
    }

    // Row labels + cells
    for (i, term_i) in terms.iter().enumerate() {
        let y = label_h + i as u32 * cell;

        // Row label
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" text-anchor="end" dominant-baseline="middle" font-size="11">{}</text>"#,
            label_w - 6,
            y + cell / 2,
            xml_escape(term_i)
        ));

        for (j, term_j) in terms.iter().enumerate() {
            let x = label_w + j as u32 * cell;
            let cos = if i == j {
                1.0 // self-similarity
            } else {
                lookup_cosine(&lookup, term_i, term_j)
            };
            let colour = cosine_to_colour(cos);

            // Cell rectangle
            svg.push_str(&format!(
                "<rect x=\"{x}\" y=\"{y}\" width=\"{cell}\" height=\"{cell}\" \
                 fill=\"{colour}\" stroke=\"#e0e0e0\" stroke-width=\"0.5\"/>"
            ));

            // Value text inside cell
            let text_colour = if cos.abs() > 0.6 { "white" } else { "#333" };
            svg.push_str(&format!(
                r#"<text x="{}" y="{}" text-anchor="middle" dominant-baseline="middle" fill="{text_colour}" font-size="10">{:.2}</text>"#,
                x + cell / 2,
                y + cell / 2,
                cos
            ));
        }
    }

    // Legend
    let lx = label_w + cell * n as u32 + 16;
    let ly = label_h;
    let lh = cell * n as u32;
    let lw = 16u32;
    let steps = 20u32;
    let step_h = lh / steps;
    for s in 0..steps {
        let cos = 1.0 - 2.0 * s as f32 / (steps - 1) as f32;
        let colour = cosine_to_colour(cos);
        let sy = ly + s * step_h;
        svg.push_str(&format!(
            r#"<rect x="{lx}" y="{sy}" width="{lw}" height="{step_h}" fill="{colour}"/>"#
        ));
    }
    // Legend labels
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-size="10" dominant-baseline="middle">+1 aligned</text>"#,
        lx + lw + 4,
        ly + 6
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-size="10" dominant-baseline="middle">0 independent</text>"#,
        lx + lw + 4,
        ly + lh / 2
    ));
    svg.push_str(&format!(
        r#"<text x="{}" y="{}" font-size="10" dominant-baseline="middle">-1 opposed</text>"#,
        lx + lw + 4,
        ly + lh - 6
    ));

    svg.push_str("</svg>");
    svg
}

// ---------------------------------------------------------------------------
// Chord diagram
// ---------------------------------------------------------------------------

/// Render a chord diagram as standalone SVG.
///
/// Terms are placed on a circle. Arcs connect each pair, coloured by
/// relationship type: red = opposed, blue = aligned, grey = independent.
/// Arc thickness and opacity scale with relationship strength.
pub fn render_chord(analysis: &CoherenceAnalysis) -> String {
    let n = analysis.num_terms;
    if n == 0 {
        return empty_svg("No terms to visualise");
    }

    let terms = ordered_terms(analysis);
    let n = terms.len();
    let _lookup = cosine_lookup(analysis);

    let size = 500u32;
    let cx = size / 2;
    let cy = size / 2;
    let radius = 180u32;
    let label_r = radius + 24;

    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" font-family="system-ui, sans-serif" font-size="12">"#
    ));
    svg.push_str(&format!(
        r#"<rect width="{size}" height="{size}" fill="white"/>"#
    ));

    // Title
    svg.push_str(&format!(
        r#"<text x="{cx}" y="24" text-anchor="middle" font-size="14" font-weight="bold">Value Coherence (score: {:.2})</text>"#,
        analysis.coherence_score
    ));

    // Compute positions on the circle
    let positions: Vec<(f32, f32)> = (0..n)
        .map(|i| {
            let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
            let px = cx as f32 + radius as f32 * angle.cos();
            let py = cy as f32 + radius as f32 * angle.sin();
            (px, py)
        })
        .collect();

    // Draw arcs (chords) between pairs
    for rel in &analysis.pairwise {
        let i = terms.iter().position(|t| t == &rel.term_a);
        let j = terms.iter().position(|t| t == &rel.term_b);
        if let (Some(i), Some(j)) = (i, j) {
            let (x1, y1) = positions[i];
            let (x2, y2) = positions[j];
            let colour = relation_colour(rel.relation);
            let opacity = relation_opacity(rel.causal_cosine);
            let width = 1.0 + 2.5 * rel.causal_cosine.abs();

            // Quadratic Bézier through the center area for a curved chord
            let mid_x = (x1 + x2) / 2.0;
            let mid_y = (y1 + y2) / 2.0;
            // Pull control point toward center for curvature
            let ctrl_x = cx as f32 + (mid_x - cx as f32) * 0.3;
            let ctrl_y = cy as f32 + (mid_y - cy as f32) * 0.3;

            svg.push_str(&format!(
                r#"<path d="M{x1:.1},{y1:.1} Q{ctrl_x:.1},{ctrl_y:.1} {x2:.1},{y2:.1}" fill="none" stroke="{colour}" stroke-width="{width:.1}" opacity="{opacity:.2}"/>"#
            ));
        }
    }

    // Draw term nodes and labels
    for (i, term) in terms.iter().enumerate() {
        let (px, py) = positions[i];
        let angle = 2.0 * std::f32::consts::PI * i as f32 / n as f32 - std::f32::consts::FRAC_PI_2;
        let lx = cx as f32 + label_r as f32 * angle.cos();
        let ly = cy as f32 + label_r as f32 * angle.sin();

        // Node circle
        let col_node = "#424242";
        svg.push_str(&format!(
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"6\" \
             fill=\"{col_node}\" stroke=\"white\" stroke-width=\"1.5\"/>"
        ));

        // Label — anchor depends on position around the circle
        let anchor = if angle.cos() > 0.3 {
            "start"
        } else if angle.cos() < -0.3 {
            "end"
        } else {
            "middle"
        };

        svg.push_str(&format!(
            r#"<text x="{lx:.1}" y="{ly:.1}" text-anchor="{anchor}" dominant-baseline="middle" font-size="12" font-weight="500">{}</text>"#,
            xml_escape(term)
        ));
    }

    // Legend
    let legend_y = size - 60;
    let col_opposed = "#d32f2f";
    let col_indep = "#bdbdbd";
    let col_aligned = "#1565c0";
    svg.push_str(&format!(
        "<line x1=\"20\" y1=\"{legend_y}\" x2=\"40\" y2=\"{legend_y}\" \
         stroke=\"{col_opposed}\" stroke-width=\"2.5\"/>"
    ));
    svg.push_str(&format!(
        "<text x=\"44\" y=\"{legend_y}\" dominant-baseline=\"middle\" \
         font-size=\"10\">opposed</text>"
    ));
    svg.push_str(&format!(
        "<line x1=\"110\" y1=\"{legend_y}\" x2=\"130\" y2=\"{legend_y}\" \
         stroke=\"{col_indep}\" stroke-width=\"2\"/>"
    ));
    svg.push_str(&format!(
        "<text x=\"134\" y=\"{legend_y}\" dominant-baseline=\"middle\" \
         font-size=\"10\">independent</text>"
    ));
    svg.push_str(&format!(
        "<line x1=\"220\" y1=\"{legend_y}\" x2=\"240\" y2=\"{legend_y}\" \
         stroke=\"{col_aligned}\" stroke-width=\"2.5\"/>"
    ));
    svg.push_str(&format!(
        "<text x=\"244\" y=\"{legend_y}\" dominant-baseline=\"middle\" \
         font-size=\"10\">aligned</text>"
    ));

    svg.push_str("</svg>");
    svg
}

// ---------------------------------------------------------------------------
// Conversation timeline
// ---------------------------------------------------------------------------

/// Render a conversation coherence timeline as standalone SVG.
///
/// Shows:
///   - A line chart of coherence score at each turn
///   - Coloured dots per speaker
///   - Red triangle markers where new contradictions emerge
///   - Value labels at each turn showing what was introduced
///   - A shaded area under the line (green→red gradient)
pub fn render_timeline(conv: &ConversationAnalysis) -> String {
    if conv.turns.is_empty() {
        return empty_svg("No conversation to visualise");
    }

    // Colour palette — kept as variables so they don't conflict with r#""#.
    let col_bg = "#0d1117";
    let col_text = "#f0f6fc";
    let col_line = "#58a6ff";
    let col_grid = "#21262d";
    let col_label = "#8b949e";
    let col_alert = "#f85149";
    let col_value = "#58a6ff";
    let col_green = "#3fb950";
    let col_stroke = "#0d1117";
    let col_legend_text = "#c9d1d9";
    let speaker_colours = ["#a5d6ff", "#7ee787", "#d2a8ff", "#ffa657"];

    let n = conv.turns.len();
    let width = 800u32;
    let height = 400u32;
    let margin_l = 60u32;
    let margin_r = 30u32;
    let margin_t = 50u32;
    let margin_b = 140u32;
    let plot_w = width - margin_l - margin_r;
    let plot_h = height - margin_t - margin_b;

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {} {}\" \
         font-family=\"system-ui, sans-serif\" font-size=\"11\">",
        width, height
    ));
    svg.push_str(&format!(
        "<rect width=\"{width}\" height=\"{height}\" fill=\"{col_bg}\"/>"
    ));

    // Title
    let final_score = conv.final_score();
    svg.push_str(&format!(
        "<text x=\"{}\" y=\"28\" text-anchor=\"middle\" font-size=\"14\" \
         font-weight=\"bold\" fill=\"{col_text}\">Coherence Timeline \u{2014} \
         final score: {:.2}</text>",
        width / 2,
        final_score
    ));

    // Gradient definition for the area fill
    svg.push_str(&format!(
        "<defs><linearGradient id=\"areaFill\" x1=\"0\" x2=\"0\" y1=\"0\" y2=\"1\">\
         <stop offset=\"0%\" stop-color=\"{col_green}\" stop-opacity=\"0.3\"/>\
         <stop offset=\"100%\" stop-color=\"{col_alert}\" stop-opacity=\"0.05\"/>\
         </linearGradient></defs>"
    ));

    // X/Y mapping
    let x_step = if n > 1 {
        plot_w as f32 / (n - 1) as f32
    } else {
        plot_w as f32
    };
    let x = |i: usize| -> f32 { margin_l as f32 + i as f32 * x_step };
    let y = |score: f32| -> f32 { margin_t as f32 + plot_h as f32 * (1.0 - score.clamp(0.0, 1.0)) };

    // Y-axis gridlines + labels
    for &tick in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let ty = y(tick);
        svg.push_str(&format!(
            "<line x1=\"{ml}\" y1=\"{ty:.1}\" x2=\"{mr}\" y2=\"{ty:.1}\" \
             stroke=\"{col_grid}\" stroke-width=\"1\"/>",
            ml = margin_l,
            mr = width - margin_r,
        ));
        svg.push_str(&format!(
            "<text x=\"{}\" y=\"{:.1}\" text-anchor=\"end\" fill=\"{col_label}\" \
             font-size=\"10\" dominant-baseline=\"middle\">{:.2}</text>",
            margin_l - 8,
            ty,
            tick,
        ));
    }

    // Area fill under the line
    let mut area_path = format!("M{:.1},{:.1}", x(0), y(conv.turns[0].analysis.coherence_score));
    for (i, turn) in conv.turns.iter().enumerate().skip(1) {
        area_path.push_str(&format!(" L{:.1},{:.1}", x(i), y(turn.analysis.coherence_score)));
    }
    area_path.push_str(&format!(
        " L{:.1},{:.1} L{:.1},{:.1} Z",
        x(n - 1),
        y(0.0),
        x(0),
        y(0.0),
    ));
    svg.push_str(&format!(
        "<path d=\"{area_path}\" fill=\"url(#areaFill)\"/>"
    ));

    // Line
    let mut line_path = format!("M{:.1},{:.1}", x(0), y(conv.turns[0].analysis.coherence_score));
    for (i, turn) in conv.turns.iter().enumerate().skip(1) {
        line_path.push_str(&format!(" L{:.1},{:.1}", x(i), y(turn.analysis.coherence_score)));
    }
    svg.push_str(&format!(
        "<path d=\"{line_path}\" fill=\"none\" stroke=\"{col_line}\" \
         stroke-width=\"2.5\" stroke-linejoin=\"round\"/>"
    ));

    // Collect unique speakers for colour assignment
    let mut speakers: Vec<String> = Vec::new();
    for t in &conv.turns {
        if !speakers.contains(&t.speaker) {
            speakers.push(t.speaker.clone());
        }
    }

    // Dots + contradiction markers + annotations
    for (i, turn) in conv.turns.iter().enumerate() {
        let px = x(i);
        let py = y(turn.analysis.coherence_score);
        let speaker_idx = speakers.iter().position(|s| s == &turn.speaker).unwrap_or(0);
        let dot_colour = speaker_colours[speaker_idx % speaker_colours.len()];

        // Dot
        svg.push_str(&format!(
            "<circle cx=\"{px:.1}\" cy=\"{py:.1}\" r=\"5\" fill=\"{dot_colour}\" \
             stroke=\"{col_stroke}\" stroke-width=\"1.5\"/>"
        ));

        // Red triangle marker if new contradictions at this turn
        if !turn.new_contradictions.is_empty() {
            let tri_y = py - 14.0;
            svg.push_str(&format!(
                "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{col_alert}\"/>",
                px, tri_y - 8.0,
                px - 5.0, tri_y,
                px + 5.0, tri_y,
            ));
            // Count label
            let count = turn.new_contradictions.len();
            if count > 1 {
                svg.push_str(&format!(
                    "<text x=\"{px:.1}\" y=\"{:.1}\" text-anchor=\"middle\" \
                     fill=\"{col_alert}\" font-size=\"9\" font-weight=\"bold\">{count}</text>",
                    tri_y - 10.0,
                ));
            }
        }

        // Turn number below x-axis
        svg.push_str(&format!(
            "<text x=\"{px:.1}\" y=\"{}\" text-anchor=\"middle\" fill=\"{col_label}\" \
             font-size=\"9\">{i}</text>",
            margin_t as f32 + plot_h as f32 + 14.0,
        ));

        // Values introduced (rotated labels below)
        if !turn.values_introduced.is_empty() {
            let label = turn.values_introduced.join(", ");
            let label_y = margin_t as f32 + plot_h as f32 + 26.0;
            svg.push_str(&format!(
                "<text x=\"{px:.1}\" y=\"{label_y:.1}\" text-anchor=\"start\" \
                 fill=\"{col_value}\" font-size=\"9\" \
                 transform=\"rotate(45,{px:.1},{label_y:.1})\">+{}</text>",
                xml_escape(&label),
            ));
        }
    }

    // Legend: speakers
    let legend_x = margin_l as f32;
    let legend_y = height as f32 - 20.0;
    for (i, speaker) in speakers.iter().enumerate() {
        let lx = legend_x + i as f32 * 140.0;
        let colour = speaker_colours[i % speaker_colours.len()];
        svg.push_str(&format!(
            "<circle cx=\"{lx:.1}\" cy=\"{:.1}\" r=\"4\" fill=\"{colour}\"/>",
            legend_y - 2.0,
        ));
        svg.push_str(&format!(
            "<text x=\"{:.1}\" y=\"{legend_y:.1}\" fill=\"{col_legend_text}\" \
             font-size=\"10\">{}</text>",
            lx + 8.0,
            xml_escape(speaker),
        ));
    }

    // Legend: contradiction marker
    let cx = legend_x + speakers.len() as f32 * 140.0;
    svg.push_str(&format!(
        "<polygon points=\"{:.1},{:.1} {:.1},{:.1} {:.1},{:.1}\" fill=\"{col_alert}\"/>",
        cx, legend_y - 8.0,
        cx - 4.0, legend_y - 1.0,
        cx + 4.0, legend_y - 1.0,
    ));
    svg.push_str(&format!(
        "<text x=\"{:.1}\" y=\"{legend_y:.1}\" fill=\"{col_legend_text}\" \
         font-size=\"10\">new contradiction</text>",
        cx + 8.0,
    ));

    svg.push_str("</svg>");
    svg
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn empty_svg(msg: &str) -> String {
    let col = "#999";
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 300 60\" \
         font-family=\"sans-serif\"><text x=\"150\" y=\"35\" \
         text-anchor=\"middle\" font-size=\"14\" fill=\"{col}\">\
         {msg}</text></svg>"
    )
}

/// Extract unique term names in stable order from the analysis.
fn ordered_terms(analysis: &CoherenceAnalysis) -> Vec<String> {
    let mut terms = Vec::new();
    for rel in &analysis.pairwise {
        if !terms.contains(&rel.term_a) {
            terms.push(rel.term_a.clone());
        }
        if !terms.contains(&rel.term_b) {
            terms.push(rel.term_b.clone());
        }
    }
    terms
}

/// Build a lookup map for cosine values.
fn cosine_lookup(analysis: &CoherenceAnalysis) -> Vec<(&str, &str, f32)> {
    analysis
        .pairwise
        .iter()
        .map(|r| (r.term_a.as_str(), r.term_b.as_str(), r.causal_cosine))
        .collect()
}

/// Look up the cosine between two terms (order-independent).
fn lookup_cosine(lookup: &[(&str, &str, f32)], a: &str, b: &str) -> f32 {
    for &(ta, tb, cos) in lookup {
        if (ta == a && tb == b) || (ta == b && tb == a) {
            return cos;
        }
    }
    0.0
}

/// Minimal XML escaping for text content.
fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coherence::{CoherenceAnalysis, PairwiseRelation, RelationType};

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
                    term_b: "transparency".into(),
                    causal_cosine: 0.2,
                    causal_distance: 1.5,
                    relation: RelationType::Independent,
                },
                PairwiseRelation {
                    term_a: "risk-aversion".into(),
                    term_b: "transparency".into(),
                    causal_cosine: -0.1,
                    causal_distance: 2.0,
                    relation: RelationType::Independent,
                },
            ],
            contradictions: vec![],
            redundancies: vec![],
            coherence_score: 0.72,
            num_terms: 3,
            num_unresolved: 0,
        }
    }

    #[test]
    fn heatmap_is_valid_svg() {
        let svg = render_heatmap(&sample_analysis());
        assert!(svg.starts_with("<svg"), "should start with <svg");
        assert!(svg.ends_with("</svg>"), "should end with </svg>");
        assert!(svg.contains("innovation"), "should contain term names");
        assert!(svg.contains("risk-aversion"));
        assert!(svg.contains("transparency"));
    }

    #[test]
    fn chord_is_valid_svg() {
        let svg = render_chord(&sample_analysis());
        assert!(svg.starts_with("<svg"), "should start with <svg");
        assert!(svg.ends_with("</svg>"));
        assert!(svg.contains("innovation"));
        assert!(svg.contains("#d32f2f"), "should contain red for opposed pair");
    }

    #[test]
    fn empty_analysis_produces_valid_svg() {
        let empty = CoherenceAnalysis {
            pairwise: vec![],
            contradictions: vec![],
            redundancies: vec![],
            coherence_score: 1.0,
            num_terms: 0,
            num_unresolved: 0,
        };
        let svg = render_heatmap(&empty);
        assert!(svg.contains("No terms"));
        let svg = render_chord(&empty);
        assert!(svg.contains("No terms"));
    }

    #[test]
    fn cosine_colour_extremes() {
        let red = cosine_to_colour(-1.0);
        assert_eq!(red, "#d32f2f", "cos=-1 should be red");
        let blue = cosine_to_colour(1.0);
        assert_eq!(blue, "#1565c0", "cos=+1 should be blue");
    }

    #[test]
    fn cosine_colour_neutral() {
        let grey = cosine_to_colour(0.0);
        assert_eq!(grey, "#f5f5f5", "cos=0 should be light grey");
    }

    #[test]
    fn xml_escape_handles_special_chars() {
        assert_eq!(xml_escape("a<b>c&d"), "a&lt;b&gt;c&amp;d");
    }

    #[test]
    fn heatmap_cell_count() {
        let svg = render_heatmap(&sample_analysis());
        // 3 terms → 9 cells (3×3 matrix)
        let cell_count = svg.matches("<rect x=").count();
        // 9 cells + 1 background + legend rects
        assert!(cell_count >= 9, "should have at least 9 cell rects, got {cell_count}");
    }
}
