use std::fs;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use ed25519_dalek::SigningKey;
use zeroize::Zeroize;

use got_attest::{assemble_and_sign, merkle_root, verify};
use got_core::geometry::CausalGeometry;
use got_core::{
    DirectionalDrift, GeometricAttestation, InnerProduct, LayerActivation, Precision,
    UnembeddingMatrix, SCHEMA_VERSION,
};
use got_probe::{
    expected_calibration_error, read_probe, read_probe_checked, train_probe,
    train_probe_calibrated, ProbeSet,
};

// ---------------------------------------------------------------------------
// CLI definition
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "got-cli",
    version,
    about = "Geometry of Trust — attestation tool"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate an Ed25519 keypair.
    Keygen {
        /// Output path for the secret key (public key is written to <path>.pub).
        #[arg(long)]
        output: PathBuf,
    },

    /// Train probes from labelled activations.
    Train {
        /// Path to activations file (.gotact binary).
        #[arg(long)]
        activations: PathBuf,
        /// Path to labels file (one `0` or `1` per line, matching activation order).
        #[arg(long)]
        labels: PathBuf,
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Which layer to train probes for.
        #[arg(long)]
        layer: usize,
        /// Dimension name for the probe.
        #[arg(long, default_value = "unnamed")]
        dimension: String,
        /// Learning rate.
        #[arg(long, default_value = "0.001")]
        lr: f32,
        /// Number of training epochs.
        #[arg(long, default_value = "100")]
        epochs: usize,
        /// Regularisation epsilon for causal geometry.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
        /// Optional path to held-out validation labels for Platt calibration.
        /// Same format as --labels. When provided, trains calibrated probes.
        #[arg(long)]
        validation_labels: Option<PathBuf>,
        /// Learning rate for Platt scaling (only used with --validation-labels).
        #[arg(long, default_value = "0.01")]
        platt_lr: f32,
        /// Number of epochs for Platt scaling (only used with --validation-labels).
        #[arg(long, default_value = "200")]
        platt_epochs: usize,
        /// Output path for trained probe set.
        #[arg(long)]
        output: PathBuf,
    },

    /// Produce a signed attestation from activations and trained probes.
    Attest {
        /// Path to activations file (.gotact binary).
        #[arg(long)]
        activations: PathBuf,
        /// Paths to probe set files (one per layer).
        #[arg(long, num_args = 1..)]
        probes: Vec<PathBuf>,
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Path to Ed25519 secret key.
        #[arg(long)]
        key: PathBuf,
        /// Model identifier string.
        #[arg(long, default_value = "unknown-model")]
        model_id: String,
        /// Corpus version string.
        #[arg(long, default_value = "unversioned")]
        corpus_version: String,
        /// Regularisation epsilon.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
        /// Optional fixed timestamp (Unix seconds) for reproducibility.
        #[arg(long)]
        timestamp: Option<u64>,
        /// Path to directory containing weight shard files (for Merkle root).
        /// If omitted, model_hash is zeroed.
        #[arg(long)]
        shards: Option<PathBuf>,
        /// Path to parent attestation JSON (for chained attestations).
        #[arg(long)]
        chain_parent: Option<PathBuf>,
        /// Path to reference geometry checkpoint (.gotgeo) for drift measurement.
        #[arg(long)]
        geo_ref: Option<PathBuf>,
        /// Output path for attestation JSON.
        #[arg(long)]
        output: PathBuf,
    },

    /// Verify a signed attestation.
    Verify {
        /// Path to attestation JSON.
        #[arg(long)]
        attestation: PathBuf,
        /// Path to Ed25519 public key.
        #[arg(long)]
        pubkey: PathBuf,
    },

    /// Save a geometry checkpoint (.gotgeo) for drift tracking.
    Checkpoint {
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Regularisation epsilon.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
        /// Output path for the geometry checkpoint.
        #[arg(long)]
        output: PathBuf,
    },

    /// Produce a calibration report for trained probes against labelled activations.
    CalibrationReport {
        /// Path to activations file (.gotact binary).
        #[arg(long)]
        activations: PathBuf,
        /// Path to labels file (one `0` or `1` per line).
        #[arg(long)]
        labels: PathBuf,
        /// Path to probe set file.
        #[arg(long)]
        probes: PathBuf,
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Regularisation epsilon.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
        /// Number of calibration bins.
        #[arg(long, default_value = "10")]
        bins: usize,
    },

    /// Compute the geometry drift between a reference checkpoint and a current model.
    Drift {
        /// Path to reference geometry checkpoint (.gotgeo).
        #[arg(long)]
        reference: PathBuf,
        /// Path to current unembedding matrix file (.gotue binary).
        #[arg(long)]
        current: PathBuf,
        /// Regularisation epsilon.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
    },

    /// Issue a signed agent certificate (PKI).
    IssueCert {
        /// Path to the CA secret key (32-byte Ed25519).
        #[arg(long)]
        ca_key: PathBuf,
        /// Path to the subject's public key (32-byte Ed25519).
        #[arg(long)]
        subject_pubkey: PathBuf,
        /// Human-readable name for the subject.
        #[arg(long)]
        subject_name: String,
        /// Roles granted to the subject (comma-separated).
        #[arg(long, value_delimiter = ',')]
        roles: Vec<String>,
        /// Validity duration in days from now.
        #[arg(long, default_value = "365")]
        validity_days: u64,
        /// Maximum geometry drift the subject may accept.
        #[arg(long, default_value = "0.05")]
        max_drift: f32,
        /// Expected model hash (64-char hex). Optional.
        #[arg(long)]
        model_hash: Option<String>,
        /// Output path for the certificate JSON.
        #[arg(long)]
        output: PathBuf,
    },

    /// Revoke a certificate by adding it to a CRL.
    RevokeCert {
        /// Path to the CA secret key.
        #[arg(long)]
        ca_key: PathBuf,
        /// Path to the certificate JSON to revoke.
        #[arg(long)]
        cert: PathBuf,
        /// Reason for revocation.
        #[arg(long, default_value = "unspecified")]
        reason: String,
        /// Path to existing CRL JSON to append to (optional; creates new CRL if absent).
        #[arg(long)]
        existing_crl: Option<PathBuf>,
        /// Next CRL update interval in days.
        #[arg(long, default_value = "30")]
        next_update_days: u64,
        /// Output path for the CRL JSON.
        #[arg(long)]
        output: PathBuf,
    },

    /// Analyse a value system for geometric coherence / contradictions.
    CoherenceCheck {
        /// Path to embeddings JSON: { "term": [f32, ...], ... }.
        #[arg(long)]
        embeddings: PathBuf,
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Regularisation epsilon.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
        /// Comma-separated list of value terms to analyse.
        /// If omitted, all terms in the embeddings file are used.
        #[arg(long, value_delimiter = ',')]
        values: Option<Vec<String>>,
        /// Causal cosine threshold for antonym detection (default: -0.5).
        #[arg(long, default_value = "-0.5")]
        antonym_threshold: f32,
        /// Causal cosine threshold for synonym detection (default: 0.8).
        #[arg(long, default_value = "0.8")]
        synonym_threshold: f32,
        /// Output format: "text", "json", "svg-heatmap", or "svg-chord" (default: text).
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Compare value geometry between two models.
    Compare {
        /// Path to first unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding_a: PathBuf,
        /// Path to second unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding_b: PathBuf,
        /// Optional path to probe set file (for probe-projected distance).
        #[arg(long)]
        probes: Option<PathBuf>,
        /// Regularisation epsilon for causal geometry.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
    },

    /// Report manifold collapse / effective value dimensionality.
    CollapseReport {
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Path to probe set file (JSON).
        #[arg(long)]
        probes: PathBuf,
        /// Regularisation epsilon for causal geometry.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
    },

    /// Compute value-ordering coherence scores for activations.
    Coherence {
        /// Path to activations file (.gotact binary).
        #[arg(long)]
        activations: PathBuf,
        /// Path to unembedding matrix file (.gotue binary).
        #[arg(long)]
        unembedding: PathBuf,
        /// Path to value-ordering constraints JSON file.
        #[arg(long)]
        ordering: PathBuf,
        /// Which layer to analyse.
        #[arg(long)]
        layer: usize,
        /// Sharpness parameter α (default 1.0). Higher = more decisive scoring.
        #[arg(long, default_value = "1.0")]
        sharpness: f32,
        /// Regularisation epsilon for causal geometry.
        #[arg(long, default_value = "0.000001")]
        epsilon: f32,
    },

    /// Perform a key rotation ceremony with mutual cross-signatures.
    RotateKey {
        /// Path to the old (current) secret key.
        #[arg(long)]
        old_key: PathBuf,
        /// Path to the new secret key.
        #[arg(long)]
        new_key: PathBuf,
        /// Path to CA secret key (to issue new certificate for the new key).
        #[arg(long)]
        ca_key: PathBuf,
        /// Validity days for the new certificate.
        #[arg(long, default_value = "365")]
        validity_days: u64,
        /// Subject name for the new certificate.
        #[arg(long)]
        subject_name: String,
        /// Roles for the new certificate (comma-separated).
        #[arg(long, value_delimiter = ',')]
        roles: Vec<String>,
        /// Maximum geometry drift for the new certificate.
        #[arg(long, default_value = "0.05")]
        max_drift: f32,
        /// Output path for the rotation JSON.
        #[arg(long)]
        output: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Keygen { output } => cmd_keygen(output),
        Command::Train {
            activations,
            labels,
            unembedding,
            layer,
            dimension,
            lr,
            epochs,
            epsilon,
            validation_labels,
            platt_lr,
            platt_epochs,
            output,
        } => cmd_train(
            activations,
            labels,
            unembedding,
            layer,
            &dimension,
            lr,
            epochs,
            epsilon,
            validation_labels,
            platt_lr,
            platt_epochs,
            output,
        ),
        Command::Attest {
            activations,
            probes,
            unembedding,
            key,
            model_id,
            corpus_version,
            epsilon,
            timestamp,
            shards,
            chain_parent,
            geo_ref,
            output,
        } => cmd_attest(
            activations,
            probes,
            unembedding,
            key,
            &model_id,
            &corpus_version,
            epsilon,
            timestamp,
            shards,
            chain_parent,
            geo_ref,
            output,
        ),
        Command::Verify {
            attestation,
            pubkey,
        } => cmd_verify(attestation, pubkey),
        Command::Checkpoint {
            unembedding,
            epsilon,
            output,
        } => cmd_checkpoint(unembedding, epsilon, output),
        Command::CalibrationReport {
            activations,
            labels,
            probes,
            unembedding,
            epsilon,
            bins,
        } => cmd_calibration_report(activations, labels, probes, unembedding, epsilon, bins),
        Command::Drift {
            reference,
            current,
            epsilon,
        } => cmd_drift(reference, current, epsilon),
        Command::IssueCert {
            ca_key,
            subject_pubkey,
            subject_name,
            roles,
            validity_days,
            max_drift,
            model_hash,
            output,
        } => cmd_issue_cert(
            ca_key,
            subject_pubkey,
            &subject_name,
            roles,
            validity_days,
            max_drift,
            model_hash,
            output,
        ),
        Command::RevokeCert {
            ca_key,
            cert,
            reason,
            existing_crl,
            next_update_days,
            output,
        } => cmd_revoke_cert(
            ca_key,
            cert,
            &reason,
            existing_crl,
            next_update_days,
            output,
        ),
        Command::CoherenceCheck {
            embeddings,
            unembedding,
            epsilon,
            values,
            antonym_threshold,
            synonym_threshold,
            format,
        } => cmd_coherence_check(
            embeddings,
            unembedding,
            epsilon,
            values,
            antonym_threshold,
            synonym_threshold,
            &format,
        ),
        Command::Compare {
            unembedding_a,
            unembedding_b,
            probes,
            epsilon,
        } => cmd_compare(unembedding_a, unembedding_b, probes, epsilon),
        Command::CollapseReport {
            unembedding,
            probes,
            epsilon,
        } => cmd_collapse_report(unembedding, probes, epsilon),
        Command::Coherence {
            activations,
            unembedding,
            ordering,
            layer,
            sharpness,
            epsilon,
        } => cmd_coherence(activations, unembedding, ordering, layer, sharpness, epsilon),
        Command::RotateKey {
            old_key,
            new_key,
            ca_key,
            validity_days,
            subject_name,
            roles,
            max_drift,
            output,
        } => cmd_rotate_key(
            old_key,
            new_key,
            ca_key,
            validity_days,
            &subject_name,
            roles,
            max_drift,
            output,
        ),
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_keygen(output: PathBuf) -> Result<()> {
    let mut rng = rand::rngs::OsRng;
    let key = SigningKey::generate(&mut rng);

    // Write secret key with restrictive permissions (owner-only read/write)
    {
        let mut key_bytes = key.to_bytes();
        #[cfg(unix)]
        {
            use std::io::Write;
            let mut f = fs::OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .mode(0o600)
                .open(&output)
                .context("failed to create secret key file")?;
            f.write_all(&key_bytes)
                .context("failed to write secret key")?;
        }
        #[cfg(not(unix))]
        {
            fs::write(&output, &key_bytes).context("failed to write secret key")?;
        }
        key_bytes.zeroize();
    }

    let pub_path = output.with_extension("pub");
    fs::write(&pub_path, key.verifying_key().to_bytes()).context("failed to write public key")?;

    println!("Secret key: {}", output.display());
    println!("Public key: {}", pub_path.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_train(
    activations_path: PathBuf,
    labels_path: PathBuf,
    unembedding_path: PathBuf,
    layer: usize,
    dimension: &str,
    lr: f32,
    epochs: usize,
    epsilon: f32,
    validation_labels_path: Option<PathBuf>,
    platt_lr: f32,
    platt_epochs: usize,
    output: PathBuf,
) -> Result<()> {
    let all_activations = load_activations(&activations_path)?;
    let u = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&u, epsilon);

    println!(
        "Geometry: d={}, rank={}",
        geometry.hidden_dim(),
        if geometry.is_positive_definite() {
            "full"
        } else {
            "deficient (regularised)"
        }
    );

    // Filter activations for the target layer
    let layer_acts: Vec<&LayerActivation> = all_activations
        .iter()
        .filter(|a| a.layer == layer)
        .collect();

    if layer_acts.is_empty() {
        bail!("no activations found for layer {layer}");
    }

    // Load labels
    let labels_text = fs::read_to_string(&labels_path).context("failed to read labels")?;
    let mut labels = Vec::new();
    for line in labels_text.lines().filter(|l| !l.trim().is_empty()) {
        match line.trim() {
            "1" | "true" => labels.push(true),
            "0" | "false" => labels.push(false),
            other => bail!("invalid label: {other}"),
        }
    }

    if labels.len() != layer_acts.len() {
        bail!(
            "label count ({}) does not match activation count ({}) for layer {layer}",
            labels.len(),
            layer_acts.len()
        );
    }

    let training_data: Vec<(Vec<f32>, bool)> = layer_acts
        .iter()
        .zip(labels.iter())
        .map(|(act, &label)| (act.values.clone(), label))
        .collect();

    let probe = if let Some(val_path) = validation_labels_path {
        // Load validation labels and activations for Platt calibration.
        let val_text = fs::read_to_string(&val_path).context("failed to read validation labels")?;
        let mut val_labels = Vec::new();
        for line in val_text.lines().filter(|l| !l.trim().is_empty()) {
            match line.trim() {
                "1" | "true" => val_labels.push(true),
                "0" | "false" => val_labels.push(false),
                other => bail!("invalid validation label: {other}"),
            }
        }
        // Use training_data for weights, validation data from the SAME activations
        // but with a separate label set. Caller is responsible for disjoint sets.
        if val_labels.len() != layer_acts.len() {
            bail!(
                "validation label count ({}) does not match activation count ({}) for layer {layer}",
                val_labels.len(),
                layer_acts.len()
            );
        }
        let validation_data: Vec<(Vec<f32>, bool)> = layer_acts
            .iter()
            .zip(val_labels.iter())
            .map(|(act, &label)| (act.values.clone(), label))
            .collect();

        println!(
            "Training calibrated probe '{}' on {} samples (+ {} validation), layer {}, lr={}, epochs={}, platt_lr={}, platt_epochs={}",
            dimension,
            training_data.len(),
            validation_data.len(),
            layer,
            lr,
            epochs,
            platt_lr,
            platt_epochs
        );

        let p = train_probe_calibrated(
            &training_data,
            &validation_data,
            &geometry,
            dimension,
            lr,
            epochs,
            platt_lr,
            platt_epochs,
        )
        .context("calibrated probe training failed")?;
        println!(
            "  platt_scale = {}, platt_shift = {}",
            p.platt_scale, p.platt_shift
        );
        p
    } else {
        println!(
            "Training probe '{}' on {} samples, layer {}, lr={}, epochs={}",
            dimension,
            training_data.len(),
            layer,
            lr,
            epochs
        );

        train_probe(&training_data, &geometry, dimension, lr, epochs)
            .context("probe training failed")?
    };

    let probe_set = ProbeSet {
        probes: vec![probe],
        version: "v0.1.0".to_string(),
        corpus_version: "unversioned".to_string(),
        layer,
        geometry_hash: Some(geometry.geometry_hash()),
        max_drift: None,
        max_directional_drift: None,
    };

    let encoded = serde_json::to_vec_pretty(&probe_set).context("failed to serialise probes")?;
    fs::write(&output, encoded).context("failed to write probes")?;
    println!("Probes written to {}", output.display());
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_attest(
    activations_path: PathBuf,
    probe_paths: Vec<PathBuf>,
    unembedding_path: PathBuf,
    key_path: PathBuf,
    model_id: &str,
    corpus_version: &str,
    epsilon: f32,
    timestamp: Option<u64>,
    shards_dir: Option<PathBuf>,
    chain_parent_path: Option<PathBuf>,
    geo_ref_path: Option<PathBuf>,
    output: PathBuf,
) -> Result<()> {
    let all_activations = load_activations(&activations_path)?;
    let u = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&u, epsilon);

    println!(
        "Geometry: d={}, rank={}",
        geometry.hidden_dim(),
        if geometry.is_positive_definite() {
            "full"
        } else {
            "deficient (regularised)"
        }
    );

    // Load all probe sets
    let mut probe_sets = Vec::new();
    for p in &probe_paths {
        let data = fs::read(p).with_context(|| format!("failed to read {}", p.display()))?;
        let ps: ProbeSet = serde_json::from_slice(&data)
            .with_context(|| format!("failed to parse {}", p.display()))?;
        probe_sets.push(ps);
    }

    // Load reference geometry early so it can be used for drift-checked probe reads.
    let ref_geometry = match geo_ref_path.as_ref() {
        Some(ref_path) => Some(load_geometry_checkpoint(ref_path)?),
        None => None,
    };

    // Run probes per layer
    let mut layer_readings: Vec<Vec<f32>> = Vec::new();
    let mut all_confidences: Vec<f32> = Vec::new();
    let mut all_coverage: Vec<bool> = Vec::new();
    let mut combined_probe_version = String::new();

    for ps in &probe_sets {
        // Find activation for this layer (use first token position found)
        let layer_act = all_activations
            .iter()
            .find(|a| a.layer == ps.layer)
            .with_context(|| format!("no activations for layer {}", ps.layer))?;

        let mut readings = Vec::new();
        for probe in &ps.probes {
            // Use drift-checked reads when the probe set has geometry tracking,
            // falling back to basic read_probe for legacy probe sets.
            let (raw, conf, flag) = if ps.geometry_hash.is_some() || ps.max_drift.is_some() {
                let reference = ref_geometry.as_ref().unwrap_or(&geometry);
                read_probe_checked(probe, ps, &layer_act.values, &geometry, reference)
                    .context("probe read failed (geometry drift or hash mismatch?)")?
            } else {
                read_probe(probe, &layer_act.values, &geometry).context("probe read failed")?
            };
            readings.push(raw);
            all_confidences.push(conf);
            all_coverage.push(flag);
        }
        layer_readings.push(readings);

        if !combined_probe_version.is_empty() {
            combined_probe_version.push('+');
        }
        combined_probe_version.push_str(&ps.version);
    }

    // Input hash: SHA-256 of the raw activations file (covers model ID, precision,
    // and all activation data — deterministic for identical extraction runs).
    let act_bytes =
        fs::read(&activations_path).context("failed to read activations for hashing")?;
    let input_hash = got_core::sha256(&act_bytes);

    // Model hash from weight shards.
    let model_hash = if let Some(dir) = shards_dir {
        let mut shard_files: Vec<_> = fs::read_dir(&dir)
            .context("failed to read shards directory")?
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_file()).unwrap_or(false))
            .collect();
        shard_files.sort_by_key(|e| e.file_name());

        let mut shards = Vec::new();
        for entry in &shard_files {
            let bytes = fs::read(entry.path())
                .with_context(|| format!("failed to read shard {}", entry.path().display()))?;
            shards.push(bytes);
        }

        Some(merkle_root(&shards))
    } else {
        None // no --shards provided
    };

    let ts = match timestamp {
        Some(t) => t,
        None => SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .context("system clock before UNIX epoch")?
            .as_secs(),
    };

    let inner_product = if geometry.is_positive_definite() {
        InnerProduct::Causal
    } else {
        InnerProduct::CausalRegularised { epsilon }
    };

    // --- Chaining support ---
    // `is_chained` controls whether we populate parent_attestation_hash and
    // geometry_hash; the schema version itself is a single constant.
    let is_chained = chain_parent_path.is_some() || geo_ref_path.is_some();
    let schema_version = SCHEMA_VERSION;

    let (parent_attestation_hash, parent_sequence_number) =
        if let Some(ref parent_path) = chain_parent_path {
            let parent_json =
                fs::read_to_string(parent_path).context("failed to read parent attestation")?;
            let parent: GeometricAttestation = serde_json::from_str(&parent_json)
                .context("failed to parse parent attestation JSON")?;
            let h = got_attest::attestation_hash(&parent)
                .context("failed to hash parent attestation")?;
            (Some(h), Some(parent.sequence_number))
        } else {
            (None, None)
        };

    let (geo_hash, geo_drift) = if let Some(ref ref_geo) = ref_geometry {
        let drift = geometry
            .drift_from(ref_geo)
            .context("drift computation failed")?;
        (Some(geometry.geometry_hash()), Some(drift))
    } else if is_chained {
        // Chained without an explicit reference: record geometry hash, drift = 0.0.
        (Some(geometry.geometry_hash()), Some(0.0))
    } else {
        (None, None)
    };

    // Compute per-probe directional drifts (Phase 13 adversarial hardening).
    let directional_drifts: Vec<DirectionalDrift> = if let Some(ref ref_geo) = ref_geometry {
        probe_sets
            .iter()
            .flat_map(|ps| &ps.probes)
            .filter_map(|probe| {
                geometry
                    .directional_drift(ref_geo, &probe.weights)
                    .ok()
                    .map(|drift| DirectionalDrift {
                        probe_name: probe.dimension_name.clone(),
                        drift,
                    })
            })
            .collect()
    } else {
        vec![]
    };

    // Probe commitment: SHA-256 over sorted probe indices (Phase 13).
    let probe_commitment = if !probe_sets.is_empty() {
        let mut probe_indices: Vec<u64> =
            (0..probe_sets.iter().map(|ps| ps.probes.len()).sum::<usize>() as u64).collect();
        probe_indices.sort();
        let commitment_bytes: Vec<u8> = probe_indices
            .iter()
            .flat_map(|idx| idx.to_le_bytes())
            .collect();
        Some(got_core::sha256(&commitment_bytes))
    } else {
        None
    };

    let attestation = GeometricAttestation {
        schema_version,
        model_id: model_id.to_string(),
        model_hash,
        precision: Precision::Fp32,
        inner_product,
        input_hash,
        timestamp: ts,
        corpus_version: corpus_version.to_string(),
        probe_version: combined_probe_version,
        layer_readings,
        confidence: all_confidences,
        coverage_flags: all_coverage.clone(),
        divergence_flag: all_coverage.iter().any(|&f| f),
        parent_attestation_hash,
        geometry_hash: geo_hash,
        geometry_drift: geo_drift,
        causal_scores: vec![],
        intervention_delta: None,
        causal_flag: None,
        sequence_number: parent_sequence_number.map_or(0, |s| s + 1),
        directional_drifts,
        probe_commitment,
        density_reading: None,
        curvature_reading: None,
        domain_scope_declaration: None,
        signature: [0u8; 64],
    };

    // Sign — load key material, use it, then zeroize
    let mut key_bytes = fs::read(&key_path).context("failed to read signing key")?;
    let key_slice: &[u8] = &key_bytes;
    let mut key_array: [u8; 32] = key_slice
        .try_into()
        .context("signing key must be exactly 32 bytes")?;
    let signing_key = SigningKey::from_bytes(&key_array);
    key_array.zeroize();
    key_bytes.zeroize();

    let signed =
        assemble_and_sign(attestation, &signing_key).context("attestation signing failed")?;

    let json = serde_json::to_string_pretty(&signed).context("JSON serialisation failed")?;
    fs::write(&output, json).context("failed to write attestation")?;

    println!("Attestation written to {}", output.display());
    println!("Reproduce with identical weights + precision + probes + input to verify.");
    Ok(())
}

fn cmd_verify(attestation_path: PathBuf, pubkey_path: PathBuf) -> Result<()> {
    let json = fs::read_to_string(&attestation_path).context("failed to read attestation")?;
    let attestation: GeometricAttestation =
        serde_json::from_str(&json).context("failed to parse attestation JSON")?;

    let pk_bytes = fs::read(&pubkey_path).context("failed to read public key")?;
    let pk_array: [u8; 32] = pk_bytes
        .as_slice()
        .try_into()
        .context("public key must be exactly 32 bytes")?;
    let verifying_key =
        ed25519_dalek::VerifyingKey::from_bytes(&pk_array).context("invalid public key")?;

    match verify(&attestation, &verifying_key) {
        Ok(()) => {
            println!("VALID — signature verifies.");
            println!("  model_id:       {}", attestation.model_id);
            println!("  schema_version: {}", attestation.schema_version);
            println!("  precision:      {:?}", attestation.precision);
            println!("  inner_product:  {:?}", attestation.inner_product);
            println!("  timestamp:      {}", attestation.timestamp);
            println!("  layers:         {}", attestation.layer_readings.len());
            let total_dims: usize = attestation.layer_readings.iter().map(|l| l.len()).sum();
            println!("  dimensions:     {total_dims}");
            let flagged: usize = attestation.coverage_flags.iter().filter(|&&f| f).count();
            println!(
                "  coverage_flags: {flagged}/{} flagged",
                attestation.coverage_flags.len()
            );
            println!("  divergence:     {}", attestation.divergence_flag);
            // Chaining info
            if let Some(ref parent_hash) = attestation.parent_attestation_hash {
                let hex: String = parent_hash.iter().map(|b| format!("{b:02x}")).collect();
                println!("  parent_hash:    {hex}");
            }
            if let Some(ref geo_hash) = attestation.geometry_hash {
                let hex: String = geo_hash.iter().map(|b| format!("{b:02x}")).collect();
                println!("  geometry_hash:  {hex}");
            }
            if let Some(drift) = attestation.geometry_drift {
                println!("  geometry_drift: {drift:.6}");
            }
        }
        Err(e) => {
            bail!("INVALID — {e}");
        }
    }
    Ok(())
}

fn cmd_checkpoint(unembedding_path: PathBuf, epsilon: f32, output: PathBuf) -> Result<()> {
    let u = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&u, epsilon);
    save_geometry_checkpoint(&geometry, &output)?;
    let hash = geometry.geometry_hash();
    let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect();
    println!("Geometry checkpoint written to {}", output.display());
    println!("  hidden_dim:     {}", geometry.hidden_dim());
    println!("  geometry_hash:  {hex}");
    Ok(())
}

fn cmd_calibration_report(
    activations_path: PathBuf,
    labels_path: PathBuf,
    probes_path: PathBuf,
    unembedding_path: PathBuf,
    epsilon: f32,
    bins: usize,
) -> Result<()> {
    let all_activations = load_activations(&activations_path)?;
    let u = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&u, epsilon);

    let data = fs::read(&probes_path)
        .with_context(|| format!("failed to read {}", probes_path.display()))?;
    let probe_set: ProbeSet = serde_json::from_slice(&data)
        .with_context(|| format!("failed to parse {}", probes_path.display()))?;

    let layer_acts: Vec<&LayerActivation> = all_activations
        .iter()
        .filter(|a| a.layer == probe_set.layer)
        .collect();

    if layer_acts.is_empty() {
        bail!("no activations found for layer {}", probe_set.layer);
    }

    let labels_text = fs::read_to_string(&labels_path).context("failed to read labels")?;
    let mut labels = Vec::new();
    for line in labels_text.lines().filter(|l| !l.trim().is_empty()) {
        match line.trim() {
            "1" | "true" => labels.push(true),
            "0" | "false" => labels.push(false),
            other => bail!("invalid label: {other}"),
        }
    }

    if labels.len() != layer_acts.len() {
        bail!(
            "label count ({}) does not match activation count ({}) for layer {}",
            labels.len(),
            layer_acts.len(),
            probe_set.layer
        );
    }

    for probe in &probe_set.probes {
        println!("Probe: {}", probe.dimension_name);
        println!(
            "  platt_scale = {}, platt_shift = {}",
            probe.platt_scale, probe.platt_shift
        );
        println!("  reliability_threshold = {}", probe.reliability_threshold);

        let mut predictions: Vec<(f32, bool)> = Vec::new();
        for (act, &label) in layer_acts.iter().zip(labels.iter()) {
            let (_raw, conf, _flag) =
                read_probe(probe, &act.values, &geometry).context("probe read failed")?;
            predictions.push((conf, label));
        }

        let ece = expected_calibration_error(&predictions, bins);
        println!("  ECE ({bins} bins) = {ece:.4}");

        // Per-bin breakdown
        println!(
            "  {:>8}  {:>8}  {:>8}  {:>5}",
            "Bin", "AvgConf", "Accuracy", "Count"
        );
        for b in 0..bins {
            let lo = b as f32 / bins as f32;
            let hi = (b + 1) as f32 / bins as f32;
            let in_bin: Vec<&(f32, bool)> = predictions
                .iter()
                .filter(|(c, _)| {
                    if b == bins - 1 {
                        *c >= lo && *c <= hi
                    } else {
                        *c >= lo && *c < hi
                    }
                })
                .collect();
            if in_bin.is_empty() {
                continue;
            }
            let n = in_bin.len();
            let avg_conf: f32 = in_bin.iter().map(|(c, _)| c).sum::<f32>() / n as f32;
            let accuracy: f32 = in_bin.iter().filter(|(_, y)| *y).count() as f32 / n as f32;
            println!(
                "  [{:.1}-{:.1})  {:>8.4}  {:>8.4}  {:>5}",
                lo, hi, avg_conf, accuracy, n
            );
        }
        println!();
    }

    Ok(())
}

fn cmd_drift(reference_path: PathBuf, current_path: PathBuf, epsilon: f32) -> Result<()> {
    let ref_geo = load_geometry_checkpoint(&reference_path)?;
    let u = load_unembedding(&current_path)?;
    let cur_geo = CausalGeometry::from_unembedding(&u, epsilon);

    let drift = cur_geo
        .drift_from(&ref_geo)
        .context("drift computation failed (dimension mismatch?)")?;

    let ref_hex: String = ref_geo
        .geometry_hash()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let cur_hex: String = cur_geo
        .geometry_hash()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();

    println!("Reference geometry: {ref_hex}");
    println!("Current geometry:   {cur_hex}");
    println!("Normalised Frobenius drift: {drift:.6}");
    if drift == 0.0 {
        println!("Geometries are identical.");
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn cmd_coherence_check(
    embeddings_path: PathBuf,
    unembedding_path: PathBuf,
    epsilon: f32,
    values: Option<Vec<String>>,
    antonym_threshold: f32,
    synonym_threshold: f32,
    format: &str,
) -> Result<()> {
    use got_incoherence::coherence::CoherenceConfig;
    use got_incoherence::embeddings::PrecomputedEmbeddings;

    // Load geometry
    let u = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&u, epsilon);

    println!(
        "Geometry: d={}, rank={}",
        geometry.hidden_dim(),
        if geometry.is_positive_definite() {
            "full"
        } else {
            "deficient (regularised)"
        }
    );

    // Load embeddings
    let emb_json =
        fs::read_to_string(&embeddings_path).context("failed to read embeddings file")?;
    let source = PrecomputedEmbeddings::from_json(&emb_json)
        .context("failed to parse embeddings JSON")?;

    // Determine which terms to analyse
    let terms: Vec<String> = if let Some(v) = values {
        v
    } else {
        // Use all terms from the embeddings file
        let map: std::collections::HashMap<String, Vec<f32>> =
            serde_json::from_str(&emb_json).context("failed to re-parse embeddings for keys")?;
        let mut keys: Vec<String> = map.into_keys().collect();
        keys.sort();
        keys
    };

    let term_refs: Vec<&str> = terms.iter().map(|s| s.as_str()).collect();

    let config = CoherenceConfig {
        antonym_threshold,
        synonym_threshold,
        severity_scale: None,
    };

    let report = got_incoherence::analyse_value_system(&term_refs, &source, &geometry, &config)
        .context("coherence analysis failed")?;

    match format {
        "json" => {
            let json = got_incoherence::report::render_json(&report.analysis)
                .context("failed to render JSON report")?;
            println!("{json}");
        }
        "svg-heatmap" => {
            print!("{}", got_incoherence::visual::render_heatmap(&report.analysis));
        }
        "svg-chord" => {
            print!("{}", got_incoherence::visual::render_chord(&report.analysis));
        }
        _ => {
            print!("{}", got_incoherence::report::render_text(&report.analysis));
            if !report.unresolved.is_empty() {
                println!("Warning: unresolved terms: {:?}", report.unresolved);
            }
        }
    }

    Ok(())
}

fn cmd_compare(
    unembedding_a_path: PathBuf,
    unembedding_b_path: PathBuf,
    probes_path: Option<PathBuf>,
    epsilon: f32,
) -> Result<()> {
    use got_core::geometry::value_alignment_distance;

    let ue_a = load_unembedding(&unembedding_a_path)?;
    let ue_b = load_unembedding(&unembedding_b_path)?;
    let geo_a = CausalGeometry::from_unembedding(&ue_a, epsilon);
    let geo_b = CausalGeometry::from_unembedding(&ue_b, epsilon);

    let (probe_refs, probe_set) = if let Some(ref path) = probes_path {
        let json = fs::read_to_string(path)
            .with_context(|| format!("failed to read probes: {path:?}"))?;
        let ps: ProbeSet = serde_json::from_str(&json).context("failed to parse probes JSON")?;
        let weights: Vec<Vec<f32>> = ps.probes.iter().map(|p| p.weights.clone()).collect();
        (Some(weights), Some(ps))
    } else {
        (None, None)
    };

    let probes_slices: Option<Vec<&[f32]>> = probe_refs
        .as_ref()
        .map(|ws| ws.iter().map(|w| w.as_slice()).collect());

    let dist = value_alignment_distance(
        &geo_a,
        &geo_b,
        probes_slices.as_deref(),
    )
    .context("alignment distance computation failed")?;

    println!("Value alignment distance");
    println!("  Model A:  {:?}", unembedding_a_path);
    println!("  Model B:  {:?}", unembedding_b_path);
    println!("  Global distance (Frobenius): {:.6}", dist.global_distance);

    if let (Some(d_v), Some(per_probe)) = (&dist.probe_projected_distance, &dist.per_probe_distances) {
        println!("  Probe-projected distance:    {:.6}", d_v);
        println!();
        if let Some(ref ps) = probe_set {
            for (i, (&d_w, probe)) in per_probe.iter().zip(ps.probes.iter()).enumerate() {
                println!("    [{i}] {}: {d_w:.6}", probe.dimension_name);
            }
        } else {
            for (i, &d_w) in per_probe.iter().enumerate() {
                println!("    [{i}] {d_w:.6}");
            }
        }
    }

    Ok(())
}

fn cmd_collapse_report(
    unembedding_path: PathBuf,
    probes_path: PathBuf,
    epsilon: f32,
) -> Result<()> {
    let ue = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&ue, epsilon);

    let probes_json = fs::read_to_string(&probes_path)
        .with_context(|| format!("failed to read probes file: {probes_path:?}"))?;
    let probe_set: ProbeSet = serde_json::from_str(&probes_json)
        .context("failed to parse probes JSON")?;

    if probe_set.probes.is_empty() {
        bail!("probe set is empty");
    }

    let weights: Vec<Vec<f32>> = probe_set.probes.iter().map(|p| p.weights.clone()).collect();
    let weight_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();

    let proj = geometry.value_projected_gram(&weight_refs)
        .context("value projection failed")?;

    let ratio = proj.dim_eff / proj.k as f32;
    let assessment = if ratio > 0.8 {
        "fully spread"
    } else if ratio > 0.4 {
        "partially collapsed"
    } else {
        "severely collapsed"
    };

    println!("Manifold collapse report (layer {})", probe_set.layer);
    println!("  Probes (k):    {}", proj.k);
    println!("  dim_eff:       {:.3}", proj.dim_eff);
    println!("  dim_eff / k:   {:.3}", ratio);
    println!("  Assessment:    {assessment}");
    println!();
    println!("  Eigenvalues of G_W (descending):");
    for (i, &ev) in proj.eigenvalues.iter().enumerate() {
        let pct = if proj.eigenvalues.iter().sum::<f32>() > 0.0 {
            100.0 * ev / proj.eigenvalues.iter().sum::<f32>()
        } else {
            0.0
        };
        println!("    λ_{i} = {ev:.6}  ({pct:.1}%)");
    }
    println!();
    println!("  Probe dimensions:");
    for p in &probe_set.probes {
        println!("    - {}", p.dimension_name);
    }

    Ok(())
}

fn cmd_coherence(
    activations_path: PathBuf,
    unembedding_path: PathBuf,
    ordering_path: PathBuf,
    layer: usize,
    sharpness: f32,
    epsilon: f32,
) -> Result<()> {
    use got_core::coherence::{conversational_coherence, ValueOrdering};

    let all_activations = load_activations(&activations_path)?;
    let ue = load_unembedding(&unembedding_path)?;
    let geometry = CausalGeometry::from_unembedding(&ue, epsilon);

    let ordering = ValueOrdering::from_json(&ordering_path)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Filter activations for the target layer
    let layer_acts: Vec<&LayerActivation> = all_activations
        .iter()
        .filter(|a| a.layer == layer)
        .collect();
    if layer_acts.is_empty() {
        bail!("no activations found for layer {layer}");
    }

    let hidden_states: Vec<Vec<f32>> = layer_acts
        .iter()
        .map(|a| a.values.clone())
        .collect();

    let report = conversational_coherence(&hidden_states, &ordering, &geometry, sharpness)
        .context("coherence computation failed")?;

    println!("Value-ordering coherence (layer {layer}, α={sharpness})");
    println!("  Positions: {}", report.per_position.len());
    println!("  Mean:      {:.4}", report.mean);
    println!("  Min:       {:.4}", report.min);
    println!("  Max:       {:.4}", report.max);
    println!();

    for (i, score) in report.per_position.iter().enumerate() {
        let marker = if *score < 0.5 { " ← VIOLATED" } else { "" };
        println!("  [{i:3}] {score:.4}{marker}");
    }

    if !report.violated_constraints.is_empty() {
        println!();
        println!("Violated constraints:");
        for (pos, label) in &report.violated_constraints {
            println!("  position {pos}: {label}");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// I/O helpers
// ---------------------------------------------------------------------------

/// Load activations from a `.gotact` binary file.
///
/// Format:
///   "GOTA" (4 bytes magic)
///   version: u16 LE
///   model_id: u32 LE len + utf8
///   precision: u8
///   hidden_dim: u32 LE
///   num_layers: u32 LE
///   num_positions: u32 LE
///   For each layer:
///     layer_index: u32 LE
///     For each position:
///       token_position: u32 LE
///       values: hidden_dim × f32 LE
fn load_activations(path: &PathBuf) -> Result<Vec<LayerActivation>> {
    let data = fs::read(path).context("failed to read activations file")?;
    let mut offset = 0;

    // Try .gotact binary format first
    if data.len() >= 4 && &data[0..4] == b"GOTA" {
        offset += 4;

        let _version = read_u16_le(&data, &mut offset)?;
        let _model_id = read_string_le(&data, &mut offset)?;
        let _precision = data[offset];
        offset += 1;
        let hidden_dim = read_u32_le(&data, &mut offset)? as usize;
        let num_layers = read_u32_le(&data, &mut offset)? as usize;
        let num_positions = read_u32_le(&data, &mut offset)? as usize;

        // Validate that declared sizes are consistent with file length.
        // Each layer has: 4 (layer_index) + num_positions * (4 + hidden_dim * 4) bytes.
        let bytes_per_position = 4usize
            .checked_add(
                hidden_dim
                    .checked_mul(4)
                    .context("overflow in gotact header")?,
            )
            .context("overflow in gotact header")?;
        let bytes_per_layer = 4usize
            .checked_add(
                num_positions
                    .checked_mul(bytes_per_position)
                    .context("overflow in gotact header")?,
            )
            .context("overflow in gotact header")?;
        let total_data_bytes = num_layers
            .checked_mul(bytes_per_layer)
            .context("overflow in gotact header")?;
        if offset
            .checked_add(total_data_bytes)
            .is_none_or(|end| end > data.len())
        {
            bail!(
                "gotact file truncated or header values invalid: need {} data bytes from offset {}, file is {} bytes",
                total_data_bytes, offset, data.len()
            );
        }

        let mut activations = Vec::with_capacity(num_layers * num_positions);
        for _ in 0..num_layers {
            let layer_index = read_u32_le(&data, &mut offset)? as usize;
            for _ in 0..num_positions {
                let token_position = read_u32_le(&data, &mut offset)? as usize;
                let mut values = Vec::with_capacity(hidden_dim);
                for _ in 0..hidden_dim {
                    values.push(read_f32_le(&data, &mut offset)?);
                }
                activations.push(LayerActivation {
                    layer: layer_index,
                    token_position,
                    values,
                });
            }
        }
        Ok(activations)
    } else {
        // Fallback: try JSON
        serde_json::from_slice(&data).context("failed to parse activations (not .gotact or JSON)")
    }
}

/// Load unembedding matrix from a `.gotue` binary file.
///
/// Format:
///   "GOTU" (4 bytes magic)
///   version: u16 LE
///   vocab_size: u32 LE
///   hidden_dim: u32 LE
///   data: V × d × f32 LE
fn load_unembedding(path: &PathBuf) -> Result<UnembeddingMatrix> {
    let data = fs::read(path).context("failed to read unembedding file")?;
    let mut offset = 0;

    if data.len() >= 4 && &data[0..4] == b"GOTU" {
        offset += 4;

        let _version = read_u16_le(&data, &mut offset)?;
        let vocab_size = read_u32_le(&data, &mut offset)? as usize;
        let hidden_dim = read_u32_le(&data, &mut offset)? as usize;

        let total = vocab_size
            .checked_mul(hidden_dim)
            .context("overflow computing vocab_size * hidden_dim")?;
        let total_bytes = total
            .checked_mul(4)
            .context("overflow computing total data bytes")?;
        if offset
            .checked_add(total_bytes)
            .is_none_or(|end| end > data.len())
        {
            bail!(
                "gotue file truncated or header values invalid: need {} data bytes from offset {}, file is {} bytes",
                total_bytes, offset, data.len()
            );
        }

        let mut values = Vec::with_capacity(total);
        for _ in 0..total {
            values.push(read_f32_le(&data, &mut offset)?);
        }

        UnembeddingMatrix::new(vocab_size, hidden_dim, values)
            .context("unembedding data length does not match vocab_size * hidden_dim")
    } else {
        // Fallback: JSON with fields { vocab_size, hidden_dim, data }
        #[derive(serde::Deserialize)]
        struct UeJson {
            vocab_size: usize,
            hidden_dim: usize,
            data: Vec<f32>,
        }
        let ue: UeJson = serde_json::from_slice(&data)
            .context("failed to parse unembedding (not .gotue or JSON)")?;
        UnembeddingMatrix::new(ue.vocab_size, ue.hidden_dim, ue.data)
            .context("unembedding JSON data length does not match vocab_size * hidden_dim")
    }
}

// Binary reading helpers — all with bounds checking

fn check_remaining(data: &[u8], offset: usize, needed: usize) -> Result<()> {
    if offset
        .checked_add(needed)
        .is_none_or(|end| end > data.len())
    {
        bail!(
            "binary file truncated: need {needed} bytes at offset {offset}, but file is {} bytes",
            data.len()
        );
    }
    Ok(())
}

fn read_u16_le(data: &[u8], offset: &mut usize) -> Result<u16> {
    check_remaining(data, *offset, 2)?;
    let v = u16::from_le_bytes(data[*offset..*offset + 2].try_into().unwrap());
    *offset += 2;
    Ok(v)
}

fn read_u32_le(data: &[u8], offset: &mut usize) -> Result<u32> {
    check_remaining(data, *offset, 4)?;
    let v = u32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(v)
}

fn read_f32_le(data: &[u8], offset: &mut usize) -> Result<f32> {
    check_remaining(data, *offset, 4)?;
    let v = f32::from_le_bytes(data[*offset..*offset + 4].try_into().unwrap());
    *offset += 4;
    Ok(v)
}

fn read_string_le(data: &[u8], offset: &mut usize) -> Result<String> {
    let len = read_u32_le(data, offset)? as usize;
    check_remaining(data, *offset, len)?;
    let s = String::from_utf8(data[*offset..*offset + len].to_vec())
        .context("invalid UTF-8 in binary string field")?;
    *offset += len;
    Ok(s)
}

// ---------------------------------------------------------------------------
// Geometry checkpoint (.gotgeo) I/O
//
// Format:
//   Magic:          4 bytes   "GOTG"
//   Version:        u16 LE    (1)
//   hidden_dim d:   u32 LE
//   geometry_hash:  32 bytes  (SHA-256 of the Gram data that follows)
//   data:           d × d × f32 LE   (row-major Gram matrix Φ)
// ---------------------------------------------------------------------------

fn save_geometry_checkpoint(geometry: &CausalGeometry, path: &PathBuf) -> Result<()> {
    let d = geometry.hidden_dim();
    let gram = geometry.gram();
    let hash = geometry.geometry_hash();

    let mut buf = Vec::with_capacity(4 + 2 + 4 + 32 + d * d * 4);
    buf.extend_from_slice(b"GOTG");
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&(d as u32).to_le_bytes());
    buf.extend_from_slice(&hash);
    for &val in gram {
        buf.extend_from_slice(&val.to_le_bytes());
    }
    fs::write(path, buf).context("failed to write geometry checkpoint")?;
    Ok(())
}

fn load_geometry_checkpoint(path: &PathBuf) -> Result<CausalGeometry> {
    let data = fs::read(path).context("failed to read geometry checkpoint")?;
    let mut offset = 0;

    check_remaining(&data, 0, 4)?;
    if &data[0..4] != b"GOTG" {
        bail!("not a .gotgeo file (bad magic)");
    }
    offset += 4;

    let _version = read_u16_le(&data, &mut offset)?;
    let d = read_u32_le(&data, &mut offset)? as usize;

    // Read stored hash (32 bytes)
    check_remaining(&data, offset, 32)?;
    let mut stored_hash = [0u8; 32];
    stored_hash.copy_from_slice(&data[offset..offset + 32]);
    offset += 32;

    // Read Gram matrix
    let total = d.checked_mul(d).context("overflow computing d*d")?;
    let total_bytes = total.checked_mul(4).context("overflow")?;
    check_remaining(&data, offset, total_bytes)?;

    let mut gram = Vec::with_capacity(total);
    for _ in 0..total {
        gram.push(read_f32_le(&data, &mut offset)?);
    }

    // Reconstruct CausalGeometry from raw Gram data
    CausalGeometry::from_raw_gram(gram, d)
        .context("geometry checkpoint Gram matrix size does not match hidden_dim squared")
}

// ---------------------------------------------------------------------------
// Key loading helper
// ---------------------------------------------------------------------------

fn load_signing_key(path: &PathBuf) -> Result<SigningKey> {
    let mut key_bytes =
        fs::read(path).with_context(|| format!("failed to read key file: {}", path.display()))?;
    let key_slice: &[u8] = &key_bytes;
    let mut key_array: [u8; 32] = key_slice
        .try_into()
        .context("key file must be exactly 32 bytes")?;
    let sk = SigningKey::from_bytes(&key_array);
    key_array.zeroize();
    key_bytes.zeroize();
    Ok(sk)
}

// ---------------------------------------------------------------------------
// PKI subcommand implementations
// ---------------------------------------------------------------------------

fn cmd_issue_cert(
    ca_key_path: PathBuf,
    subject_pubkey_path: PathBuf,
    subject_name: &str,
    roles: Vec<String>,
    validity_days: u64,
    max_drift: f32,
    model_hash_hex: Option<String>,
    output: PathBuf,
) -> Result<()> {
    use got_wire::certificate::sign_certificate;

    // Load CA signing key.
    let ca_sk = load_signing_key(&ca_key_path)?;

    // Load subject public key.
    let subject_bytes =
        fs::read(&subject_pubkey_path).context("failed to read subject public key")?;
    if subject_bytes.len() != 32 {
        bail!(
            "subject public key must be exactly 32 bytes, got {}",
            subject_bytes.len()
        );
    }
    let mut pk_arr = [0u8; 32];
    pk_arr.copy_from_slice(&subject_bytes);
    let subject_pk =
        ed25519_dalek::VerifyingKey::from_bytes(&pk_arr).context("invalid Ed25519 public key")?;

    // Parse optional model hash.
    let expected_model_hash = match model_hash_hex {
        Some(ref hex) => {
            if hex.len() != 64 {
                bail!("--model-hash must be 64 hex chars, got {}", hex.len());
            }
            let bytes: Vec<u8> = (0..64)
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i + 2], 16))
                .collect::<std::result::Result<Vec<u8>, _>>()
                .context("invalid hex in --model-hash")?;
            let mut arr = [0u8; 32];
            arr.copy_from_slice(&bytes);
            Some(arr)
        }
        None => None,
    };

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let not_after = now + validity_days * 86400;

    let cert = sign_certificate(
        subject_name,
        &subject_pk,
        roles,
        max_drift,
        expected_model_hash,
        now,
        not_after,
        &ca_sk,
    );

    let json =
        serde_json::to_string_pretty(&cert).context("failed to serialize certificate to JSON")?;
    fs::write(&output, json.as_bytes()).context("failed to write certificate")?;

    println!(
        "Certificate issued for '{}' → {}",
        subject_name,
        output.display()
    );
    println!("  Valid: {} — {}", now, not_after);
    println!("  Roles: {:?}", cert.roles);
    Ok(())
}

fn cmd_revoke_cert(
    ca_key_path: PathBuf,
    cert_path: PathBuf,
    reason: &str,
    existing_crl_path: Option<PathBuf>,
    next_update_days: u64,
    output: PathBuf,
) -> Result<()> {
    use got_wire::certificate::{
        certificate_fingerprint, sign_crl, CertificateRevocationList, RevokedEntry,
    };

    let ca_sk = load_signing_key(&ca_key_path)?;

    // Load and parse the certificate.
    let cert_json = fs::read_to_string(&cert_path).context("failed to read certificate file")?;
    let cert: got_wire::certificate::AgentCertificate =
        serde_json::from_str(&cert_json).context("failed to parse certificate JSON")?;
    let fingerprint = certificate_fingerprint(&cert);

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Start from existing CRL or create a fresh one.
    let mut entries = Vec::new();
    if let Some(ref crl_path) = existing_crl_path {
        let crl_json = fs::read_to_string(crl_path).context("failed to read existing CRL")?;
        let existing: CertificateRevocationList =
            serde_json::from_str(&crl_json).context("failed to parse existing CRL")?;
        entries = existing.entries;
    }

    entries.push(RevokedEntry {
        certificate_fingerprint: fingerprint,
        revocation_time: now,
        reason: reason.to_string(),
    });

    let next_update = now + next_update_days * 86400;
    let crl = sign_crl(entries, now, next_update, &ca_sk);

    let json = serde_json::to_string_pretty(&crl).context("failed to serialize CRL to JSON")?;
    fs::write(&output, json.as_bytes()).context("failed to write CRL")?;

    let fp_hex: String = fingerprint.iter().map(|b| format!("{b:02x}")).collect();
    println!("Certificate revoked: {}", fp_hex);
    println!("CRL written to {}", output.display());
    println!("  Total revoked entries: {}", crl.entries.len());
    Ok(())
}

fn cmd_rotate_key(
    old_key_path: PathBuf,
    new_key_path: PathBuf,
    ca_key_path: PathBuf,
    validity_days: u64,
    subject_name: &str,
    roles: Vec<String>,
    max_drift: f32,
    output: PathBuf,
) -> Result<()> {
    use got_wire::certificate::{create_rotation, sign_certificate};

    let old_sk = load_signing_key(&old_key_path)?;
    let new_sk = load_signing_key(&new_key_path)?;
    let ca_sk = load_signing_key(&ca_key_path)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Issue a new certificate for the new key.
    let new_cert = sign_certificate(
        subject_name,
        &new_sk.verifying_key(),
        roles,
        max_drift,
        None,
        now,
        now + validity_days * 86400,
        &ca_sk,
    );

    let rotation = create_rotation(&old_sk, &new_sk, new_cert, now);

    let json =
        serde_json::to_string_pretty(&rotation).context("failed to serialize rotation to JSON")?;
    fs::write(&output, json.as_bytes()).context("failed to write rotation")?;

    let old_hex: String = old_sk
        .verifying_key()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let new_hex: String = new_sk
        .verifying_key()
        .as_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("Key rotation created:");
    println!("  Old key: {}", old_hex);
    println!("  New key: {}", new_hex);
    println!("  Written to {}", output.display());
    Ok(())
}
