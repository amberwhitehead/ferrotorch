//! Text-embedding inference-dump binary for the BERT real-artifact harness.
//!
//! Companion to `scripts/verify_text_embedding_inference.py`. Loads one
//! of the pinned text-embedding mirrors from `ferrotorch/<name>` on the
//! HuggingFace Hub, runs a single forward pass on a fixed sentence,
//! mean-pools over the attention mask, L2-normalizes (if the mirror
//! requests it), and dumps the resulting embedding to disk in the same
//! `[u32 ndim][u32 × ndim shape][f32 data]` little-endian format the
//! vision / causal-LM dump examples use.
//!
//! Usage (network required for first-touch; subsequent runs use the
//! local hub cache):
//! ```text
//! cargo run -p ferrotorch-bert --release --example text_embedding_dump -- \
//!     --model all-MiniLM-L6-v2 \
//!     --sentence "The quick brown fox jumps over the lazy dog." \
//!     --output /tmp/rust_emb.bin
//! ```
//!
//! Output:
//!   * `--output <path>`: embedding tensor `[1, hidden]` in the format
//!     above.
//!   * stdout: one JSON line
//!     `{"shape":[1,H],"normalize":true,"token_ids":[...],
//!       "kept":N,"sentence":"..."}`
//!     so the Python harness can parse the verdict.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use ferrotorch_bert::{BertConfig, HfBertConfig, SentenceTransformer, load_sentence_transformer};
use ferrotorch_core::FerrotorchResult;
use ferrotorch_hub::{HubCache, hf_download_model};
use ferrotorch_tokenize::{encode, load_tokenizer};

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    sentence: Option<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut sentence: Option<String> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(
                    argv.get(i + 1)
                        .ok_or("--model needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            "--sentence" => {
                sentence = Some(
                    argv.get(i + 1)
                        .ok_or("--sentence needs a value")?
                        .clone(),
                );
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model all-MiniLM-L6-v2)")?,
        output: output.ok_or("--output is required (path to embedding .bin)")?,
        sentence,
    })
}

fn write_dump_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data length {} disagrees with shape product {}",
        data.len(),
        expected
    );
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;

    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[text_embedding_dump] repo = {repo}");

    // -- 1. Download the full bundle into the hub cache. -----------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[text_embedding_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir)
            .map(|r| r.count())
            .unwrap_or(0)
    );

    // -- 2. Parse config + tokenizer. ------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let hf_cfg = HfBertConfig::from_file(&cfg_path)?;
    let cfg = BertConfig::from_hf(&hf_cfg)?;
    eprintln!(
        "[text_embedding_dump] cfg: hidden={} layers={} heads={} vocab={}",
        cfg.hidden_size, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.vocab_size,
    );

    let tok = load_tokenizer(repo_dir.join("tokenizer.json"))?;

    // -- 3. Resolve the sentence. ----------------------------------------
    let sentence_str = if let Some(s) = args.sentence.clone() {
        s
    } else {
        let parity = repo_dir.join("_value_parity_input.txt");
        let raw = std::fs::read_to_string(&parity).map_err(|e| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!("missing parity-probe sentence {}: {e}", parity.display()),
            }
        })?;
        raw.trim_end_matches('\n').to_string()
    };
    eprintln!("[text_embedding_dump] sentence = {sentence_str:?}");

    // -- 4. Encode locally (ferrotorch-tokenize, BERT WordPiece w/ CLS/SEP).
    let raw_ids = encode(&tok, &sentence_str, /* add_special_tokens = */ true)?;
    // The tokenizer ships with `Fixed(128)` padding + `[PAD]=0`. To match
    // the Python sentence-transformers path we need to drop trailing
    // `pad_token_id` rows; the attention_mask is then a 1-prefix over
    // the kept rows. This is exactly how sentence-transformers itself
    // constructs the mask (via `tokenizer.pad_token_id`).
    let pad_id: u32 = cfg.pad_token_id as u32;
    // Find the first trailing run of pad_id. We only strip a *suffix*
    // of pad tokens — interior zeros (if any) are kept (defensive,
    // though uncommon for BERT inputs).
    let trim_to = {
        let mut end = raw_ids.len();
        while end > 0 && raw_ids[end - 1] == pad_id {
            end -= 1;
        }
        end
    };
    let input_ids: Vec<u32> = raw_ids[..trim_to].to_vec();
    eprintln!(
        "[text_embedding_dump] tokenized to {} ids (after pad-trim from {}): {:?}",
        input_ids.len(),
        raw_ids.len(),
        input_ids,
    );

    // The frozen token ids in the mirror let us catch tokenizer drift loudly.
    let frozen_path = repo_dir.join("_value_parity_token_ids.json");
    if frozen_path.exists() {
        let raw = std::fs::read_to_string(&frozen_path).map_err(|e| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!("failed reading {}: {e}", frozen_path.display()),
            }
        })?;
        let frozen = parse_u32_array(raw.trim()).map_err(|m| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!("parsing {}: {m}", frozen_path.display()),
            }
        })?;
        if frozen != input_ids {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "tokenizer mismatch: local={input_ids:?} vs frozen={frozen:?}"
                ),
            });
        }
        eprintln!("[text_embedding_dump] local encode matches frozen token_ids");
    }

    // attention_mask = [1; N] now that we've stripped pad-suffix.
    // sentence-transformers normalize=True per the `2_Normalize` module
    // in the all-MiniLM-L6-v2 repo.
    let attention_mask: Vec<u32> = vec![1u32; input_ids.len()];
    let normalize = true;

    // -- 5. Load weights and build SentenceTransformer. ------------------
    let weights_path = repo_dir.join("model.safetensors");
    let (st, drop_report): (SentenceTransformer<f32>, _) =
        load_sentence_transformer::<f32>(&weights_path, cfg, normalize, /* strict = */ false)?;
    eprintln!(
        "[text_embedding_dump] loaded weights: dropped_position_ids={} dropped_pooler={:?}",
        drop_report.dropped_position_ids, drop_report.dropped_pooler
    );

    // -- 6. Forward + pool + normalize + dump. ---------------------------
    let emb = st.encode(&input_ids, Some(&attention_mask), None)?;
    let shape = emb.shape();
    let data = emb.data()?;
    assert_eq!(shape.len(), 2, "embedding must be [1, H], got {shape:?}");

    write_dump_f32(&args.output, shape, data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing embedding to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[text_embedding_dump] wrote {} ({} bytes, shape={shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // -- 7. JSON verdict line. -------------------------------------------
    let kept: u32 = attention_mask.iter().sum();
    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"shape\":[{},{}],", shape[0], shape[1]));
    out.push_str(&format!("\"normalize\":{},", normalize));
    out.push_str(&format!("\"kept\":{},", kept));
    out.push_str("\"token_ids\":[");
    for (i, id) in input_ids.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&id.to_string());
    }
    out.push_str("],");
    // The sentence may contain double quotes; escape minimally for stdout-only consumption.
    out.push_str("\"sentence\":\"");
    for ch in sentence_str.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            other => out.push(other),
        }
    }
    out.push('"');
    out.push('}');
    println!("{out}");

    Ok(())
}

fn parse_u32_array(s: &str) -> Result<Vec<u32>, String> {
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("not a JSON array: {s:?}"))?;
    let mut out = Vec::new();
    for chunk in inner.split(',') {
        let t = chunk.trim();
        if t.is_empty() {
            continue;
        }
        let v: u32 = t.parse().map_err(|e| format!("parse {t:?}: {e}"))?;
        out.push(v);
    }
    Ok(out)
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[text_embedding_dump] error: {e}");
        std::process::exit(1);
    }
}
