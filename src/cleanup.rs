use rand::seq::SliceRandom;

use crate::folding::{bp_distance_to_target, ensemble_diversity};
use crate::io::gc_content;
use crate::structure::get_pair_map;
use crate::types::DesignResult;
use std::{collections::HashMap, fs, io, path::Path};

pub fn gc_cleanup(
    seq_in: &str,
    original_annotation: &str,
    target: &str,
    min_gc_percent: f64,
) -> String {
    assert_eq!(seq_in.len(), original_annotation.len());
    assert_eq!(seq_in.len(), target.len());

    let pair_map = get_pair_map(target);
    let annotation = original_annotation.as_bytes();
    let mut seq: Vec<u8> = seq_in.as_bytes().to_vec();
    let mut rng = rand::rng();

    let mut current_result =
        bp_distance_to_target(&String::from_utf8(seq.clone()).unwrap(), target);
    let mut current_dist = current_result.bp_distance;

    let mut pairs: Vec<(usize, usize)> = pair_map
        .iter()
        .filter_map(|(&i, &j)| (i < j).then_some((i, j)))
        .filter(|&(i, j)| annotation[i] == b'N' && annotation[j] == b'N')
        .filter(|&(i, j)| matches!((seq[i], seq[j]), (b'G', b'C') | (b'C', b'G')))
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    pairs.shuffle(&mut rng);

    let step = 200.0 / seq.len() as f64;
    let mut gc = gc_content(&String::from_utf8(seq.clone()).unwrap()).unwrap_or(0.0);

    let mut total_replacements = 0usize;

    for &(i, j) in &pairs {
        if gc - step < min_gc_percent {
            break;
        }

        let mut options = [(b'A', b'U'), (b'U', b'A')];
        options.shuffle(&mut rng);
        let (div_nested, div_pk) = ensemble_diversity(&seq_in, target);
        let ensemble_diversity_score = div_nested + div_pk;
        let mut current_div = ensemble_diversity_score;

        let mut best_trial: Option<(Vec<u8>, DesignResult)> = None;
        for (left, right) in options {
            let mut trial = seq.clone();
            trial[i] = left;
            trial[j] = right;

            let (div_nested, div_pk) = ensemble_diversity(&seq_in, target);
            let ensemble_diversity_score = div_nested + div_pk;
            let trial_result =
                bp_distance_to_target(&String::from_utf8(trial.clone()).unwrap(), target);
            if trial_result.bp_distance > current_dist {
                continue;
            }

            let replace_best = match &best_trial {
                None => true,
                Some((_, b)) => {
                    trial_result.bp_distance < b.bp_distance
                        && ensemble_diversity_score < current_div
                        || (trial_result.bp_distance == b.bp_distance && trial_result.mfe < b.mfe)
                            && ensemble_diversity_score < current_div
                }
            };
            if replace_best {
                best_trial = Some((trial, trial_result));
                current_div = ensemble_diversity_score;
            }
        }

        if let Some((trial, trial_result)) = best_trial {
            seq = trial;
            current_dist = trial_result.bp_distance;
            current_result = trial_result;
            gc -= step;
            total_replacements += 1;
        }
    }

    println!(
        "GC cleanup complete: {} GC pair(s) converted; final bp_distance={}",
        total_replacements, current_result.bp_distance
    );

    String::from_utf8(seq).expect("sequence must be valid UTF-8")
}

pub fn diversify_sequence(
    seq_in: &str,
    original_annotation: &str,
    target: &str,
    min_gc_percent: f64,
) -> String {
    assert_eq!(seq_in.len(), original_annotation.len());
    assert_eq!(seq_in.len(), target.len());

    let pair_map = get_pair_map(target);
    let annotation = original_annotation.as_bytes();
    let mut seq: Vec<u8> = seq_in.as_bytes().to_vec();
    let mut rng = rand::rng();

    let current_result = bp_distance_to_target(&String::from_utf8(seq.clone()).unwrap(), target);
    let mut current_dist = current_result.bp_distance;

    let mut pairs: Vec<(usize, usize)> = pair_map
        .iter()
        .filter_map(|(&i, &j)| (i < j).then_some((i, j)))
        .filter(|&(i, j)| annotation[i] == b'N' && annotation[j] == b'N')
        .filter(|&(i, j)| {
            matches!(
                (seq[i], seq[j]),
                (b'G', b'C') | (b'C', b'G') | (b'A', b'U') | (b'U', b'A')
            )
        })
        .collect();
    pairs.sort_unstable();
    pairs.dedup();

    pairs.shuffle(&mut rng);

    let step = 200.0 / seq.len() as f64;
    let mut gc = gc_content(&String::from_utf8(seq.clone()).unwrap()).unwrap_or(0.0);

    for &(i, j) in &pairs {
        if gc - step < min_gc_percent {
            break;
        }

        let mut options = [(b'A', b'U'), (b'U', b'A'), (b'G', b'C'), (b'C', b'G')];
        options.shuffle(&mut rng);
        let (div_nested, div_pk) = ensemble_diversity(&seq_in, target);
        let ensemble_diversity_score = div_nested + div_pk;
        let mut current_div = ensemble_diversity_score;

        let mut best_trial: Option<(Vec<u8>, DesignResult)> = None;
        for (left, right) in options {
            let mut trial = seq.clone();
            trial[i] = left;
            trial[j] = right;
            let (div_nested, div_pk) = ensemble_diversity(&seq_in, target);
            let ensemble_diversity_score = div_nested + div_pk;
            let trial_result =
                bp_distance_to_target(&String::from_utf8(trial.clone()).unwrap(), target);
            if trial_result.bp_distance > current_dist {
                continue;
            }

            let replace_best = match &best_trial {
                None => true,
                Some((_, b)) => {
                    trial_result.bp_distance < b.bp_distance
                        && ensemble_diversity_score < current_div
                        || (trial_result.bp_distance == b.bp_distance && trial_result.mfe < b.mfe)
                            && ensemble_diversity_score < current_div
                }
            };
            if replace_best {
                best_trial = Some((trial, trial_result));
                current_div = ensemble_diversity_score;
            }
        }

        if let Some((trial, trial_result)) = best_trial {
            seq = trial;
            current_dist = trial_result.bp_distance;
            //current_result = trial_result;
            gc -= step;
        }
    }

    String::from_utf8(seq).expect("sequence must be valid UTF-8")
}






pub fn flip_restriction_sites(
    seq_in: &str,
    original_annotation: &str,
    target: &str,
    min_gc_percent: f64,
    sites_path: impl AsRef<Path>,
) -> io::Result<(String, Vec<(usize, String)>)> {
    assert_eq!(seq_in.len(), original_annotation.len());
    assert_eq!(seq_in.len(), target.len());

    eprintln!("Removing restriction sites from sequence...");
    let text = fs::read_to_string(sites_path)?;
    let mut sites: Vec<Vec<u8>> = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let token = raw.split('#').next().unwrap_or("").trim();
        if token.is_empty() {
            continue;
        }
        let site: Vec<u8> = token.bytes().map(|b| b.to_ascii_uppercase()).collect();
        if !site.iter().all(|c| b"ACGU".contains(c)) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("line {}: invalid RNA restriction site '{}'", n + 1, token),
            ));
        }
        sites.push(site);
    }
    if sites.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "no restriction sites found in file",
        ));
    }
    sites.sort();
    sites.dedup();

    
    let pair_map: HashMap<usize, usize> = get_pair_map(target);
    let annotation = original_annotation.as_bytes();
    let mut seq: Vec<u8> = seq_in.to_ascii_uppercase().into_bytes();
    let mut rng = rand::rng();

    let to_str = |v: &[u8]| String::from_utf8(v.to_vec()).expect("valid UTF-8");
    let find_hits = |s: &[u8]| -> Vec<(usize, usize)> {
        let mut hits = Vec::new();
        for (si, site) in sites.iter().enumerate() {
            if site.len() > s.len() {
                continue;
            }
            for start in 0..=(s.len() - site.len()) {
                if &s[start..start + site.len()] == site.as_slice() {
                    hits.push((start, si));
                }
            }
        }
        hits
    };

    const WC_PAIRS: [(u8, u8); 4] = [(b'A', b'U'), (b'U', b'A'), (b'G', b'C'), (b'C', b'G')];
    const BASES: [u8; 4] = *b"ACGU";

    let baseline_dist = bp_distance_to_target(&to_str(&seq), target).bp_distance;
    let mut current_gc = gc_content(&to_str(&seq)).unwrap_or(0.0);

   
    let max_rounds = seq.len() * 10 + 100;
    for _ in 0..max_rounds {
        let mut hits = find_hits(&seq);
        if hits.is_empty() {
            break;
        }
        let current_hits = hits.len();
        hits.shuffle(&mut rng);

        let mut progressed = false;

        for &(start, si) in &hits {
            let end = start + sites[si].len();

            
            let mut candidates: Vec<Vec<(usize, u8)>> = Vec::new();
            for p in start..end {
                if annotation[p] != b'N' {
                    continue;
                }
                match pair_map.get(&p) {
                    Some(&q) => {
                        if annotation[q] != b'N' {
                            continue;
                        }
                        for &(l, r) in &WC_PAIRS {
                            if seq[p] != l || seq[q] != r {
                                candidates.push(vec![(p, l), (q, r)]);
                            }
                        }
                    }
                    None => {
                        for &b in &BASES {
                            if b != seq[p] {
                                candidates.push(vec![(p, b)]);
                            }
                        }
                    }
                }
            }
            candidates.shuffle(&mut rng); // random tie-breaking

            let mut best: Option<(Vec<u8>, usize, DesignResult, f64)> = None;
            for edit in candidates {
                let mut trial = seq.clone();
                for &(pos, b) in &edit {
                    trial[pos] = b;
                }

                
                let n_hits = find_hits(&trial).len();
                if n_hits >= current_hits {
                    continue;
                }

                let trial_str = to_str(&trial);
                let gc = gc_content(&trial_str).unwrap_or(0.0);
                if gc < min_gc_percent && gc < current_gc {
                    continue;
                }

                let result = bp_distance_to_target(&trial_str, target);
                if result.bp_distance > baseline_dist {
                    continue;
                }

                let better = match &best {
                    None => true,
                    Some((_, bh, br, _)) => {
                        (n_hits, result.bp_distance) < (*bh, br.bp_distance)
                            || ((n_hits, result.bp_distance) == (*bh, br.bp_distance)
                                && result.mfe < br.mfe)
                    }
                };
                if better {
                    best = Some((trial, n_hits, result, gc));
                }
            }

            if let Some((trial, _, _, gc)) = best {
                seq = trial;
                current_gc = gc;
                progressed = true;
                break; // hits have shifted: rescan
            }
        }

        if !progressed {
            break; // no valid edit for any remaining site
        }
    }

    let remaining = find_hits(&seq)
        .into_iter()
        .map(|(start, si)| (start, to_str(&sites[si])))
        .collect();

    Ok((to_str(&seq), remaining))
}