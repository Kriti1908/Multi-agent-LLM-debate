"""
results_writer.py
-----------------
Serialises all metric results to:
  1. results/metrics_full.json   – complete raw results (all granularities)
  2. results/summary_tables.txt  – human-readable summary tables
  3. results/per_model/          – one JSON per model with all its metrics
  4. results/per_value/          – one JSON per value with all its metrics
"""

import os
import json
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helper (handles tuple keys)
# ---------------------------------------------------------------------------

def _convert_keys(obj):
    """Recursively convert tuple keys to strings so json.dump works."""
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            new_key = ' | '.join(str(x) for x in k) if isinstance(k, tuple) else k
            new[new_key] = _convert_keys(v)
        return new
    elif isinstance(obj, list):
        return [_convert_keys(i) for i in obj]
    elif isinstance(obj, tuple):
        return [_convert_keys(i) for i in obj]
    elif isinstance(obj, set):
        return [_convert_keys(i) for i in sorted(obj, key=str)]
    return obj


def _dump(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(_convert_keys(obj), f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Main writer
# ---------------------------------------------------------------------------

def write_all_results(results: dict, out_dir: str = 'results'):
    """
    results dict expected keys (all optional — present ones are written):
      tc, steps, ar, strength, pb, elo, ibc, consensus,
      usa, escalation, per,
      mm_vol_agree, mm_strength_win, mm_ibc_win, mm_cmpa,
      mm_vdm, mm_mvas, mm_efficiency, mm_dpi,
      mm_strength_agree, mm_same_model,
      vdi, pb_summary, spearman_elo_vdi
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Full dump
    _dump(results, os.path.join(out_dir, 'metrics_full.json'))

    # 2. Per-model files
    _write_per_model(results, out_dir)

    # 3. Per-value files
    _write_per_value(results, out_dir)

    # 4. Summary tables
    _write_summary_tables(results, out_dir)

    logger.info("All results written to %s/", out_dir)


# ---------------------------------------------------------------------------
# Per-model breakdown
# ---------------------------------------------------------------------------

def _write_per_model(results: dict, out_dir: str):
    model_dir = os.path.join(out_dir, 'per_model')
    os.makedirs(model_dir, exist_ok=True)

    # Collect all model names across all metrics
    model_names = set()
    for key in ['tc', 'steps', 'ar', 'strength', 'pb', 'ibc', 'consensus', 'usa', 'per']:
        if key in results:
            model_names.update(results[key].keys())

    for model in model_names:
        model_data = {'model': model}

        # Base metrics
        for metric_key, label in [
            ('tc', 'token_cost'),
            ('steps', 'reasoning_steps'),
            ('ar', 'argument_result'),
            ('strength', 'argument_strength'),
            ('pb', 'persuasion_bias'),
            ('ibc', 'identity_bias_coefficient'),
            ('consensus', 'moral_consensus_objection'),
            ('usa', 'unethical_strategy'),
            ('per', 'persuasion_effectiveness'),
        ]:
            if metric_key in results and model in results[metric_key]:
                model_data[label] = results[metric_key][model]

        # ELO
        if 'elo' in results:
            elo = results['elo']
            model_data['elo_overall'] = elo.get('overall_model', {}).get(model)
            model_data['elo_delta'] = elo.get('delta_elo', {}).get(model)
            model_data['elo_by_value'] = dict(elo.get('value_ranking_by_model', {}).get(model, []))

        # Mixed metrics
        for mk, label in [
            ('mm_efficiency', 'persuasion_efficiency'),
            ('mm_dpi', 'polarization_index'),
        ]:
            if mk in results and model in results[mk]:
                model_data[label] = results[mk][model]

        # Summary
        if 'pb_summary' in results and model in results['pb_summary']:
            model_data['overall_win_rate'] = results['pb_summary'][model]

        if 'spearman_elo_vdi' in results and model in results['spearman_elo_vdi']:
            model_data['spearman_elo_vs_vdi'] = results['spearman_elo_vdi'][model]

        safe_name = model.replace('/', '_').replace(':', '_')
        _dump(model_data, os.path.join(model_dir, f'{safe_name}.json'))


# ---------------------------------------------------------------------------
# Per-value breakdown
# ---------------------------------------------------------------------------

def _write_per_value(results: dict, out_dir: str):
    value_dir = os.path.join(out_dir, 'per_value')
    os.makedirs(value_dir, exist_ok=True)

    # Collect all values from VDI
    all_values = set()
    if 'vdi' in results:
        all_values.update(results['vdi'].keys())
    # Also scan pb
    if 'pb' in results:
        for model_data in results['pb'].values():
            all_values.update(model_data.keys())

    for value in all_values:
        value_data = {'value': value}

        # VDI
        if 'vdi' in results and value in results['vdi']:
            value_data['value_dominance_index'] = results['vdi'][value]

        # Per-model breakdown for this value
        value_data['per_model'] = {}
        for metric_key, label in [
            ('tc', 'token_cost'),
            ('steps', 'reasoning_steps'),
            ('ar', 'argument_result'),
            ('strength', 'argument_strength'),
            ('pb', 'persuasion_bias'),
            ('ibc', 'identity_bias_coefficient'),
            ('consensus', 'moral_consensus_objection'),
        ]:
            if metric_key not in results:
                continue
            for model, val_dict in results[metric_key].items():
                if value in val_dict:
                    if model not in value_data['per_model']:
                        value_data['per_model'][model] = {}
                    value_data['per_model'][model][label] = val_dict[value]

        # Value dominance matrix row
        if 'mm_vdm' in results and 'matrix' in results['mm_vdm']:
            value_data['vs_other_values'] = results['mm_vdm']['matrix'].get(value, {})

        safe_name = value.replace('/', '_').replace(' ', '_')
        _dump(value_data, os.path.join(value_dir, f'{safe_name}.json'))


# ---------------------------------------------------------------------------
# Summary tables (plain text)
# ---------------------------------------------------------------------------

def _write_summary_tables(results: dict, out_dir: str):
    lines = []

    def h(title):
        lines.append('')
        lines.append('=' * 70)
        lines.append(f'  {title}')
        lines.append('=' * 70)

    def row(*cols, widths=None):
        if widths is None:
            widths = [30] + [12] * (len(cols) - 1)
        parts = []
        for i, c in enumerate(cols):
            w = widths[i] if i < len(widths) else 12
            parts.append(str(c)[:w].ljust(w))
        lines.append('  '.join(parts))

    # ── Overall model win rates ──────────────────────────────────────────────
    h('PERSUASION BIAS — Overall Model Win Rates')
    if 'pb_summary' in results:
        row('Model', 'Win Rate', 'Total Debates')
        for model, d in sorted(results['pb_summary'].items(),
                               key=lambda x: -x[1]['overall_win_rate']):
            row(model, f"{d['overall_win_rate']:.3f}", d['total_debates'])

    # ── ELO Rankings ────────────────────────────────────────────────────────
    h('ELO RANKINGS — Overall Model Ratings')
    if 'elo' in results:
        row('Model', 'ELO Rating', 'Δ-ELO')
        for model, rating in sorted(
            results['elo']['overall_model'].items(), key=lambda x: -x[1]
        ):
            delta = results['elo']['delta_elo'].get(model, 0)
            row(model, f"{rating:.1f}", f"{delta:.1f}")

    # ── Token Cost ───────────────────────────────────────────────────────────
    h('TOKEN COST — Mean Words per Debate by Model')
    if 'tc' in results:
        row('Model', 'Value', 'Mean Words/Debate', 'N Debates')
        for model in sorted(results['tc'].keys()):
            for value, d in sorted(results['tc'][model].items()):
                row(model, value[:28], f"{d['mean_words_per_debate']:.0f}", d['num_debates'])

    # ── Value Dominance Index ────────────────────────────────────────────────
    h('VALUE DOMINANCE INDEX — Global Win Rates per Value')
    if 'vdi' in results:
        row('Value', 'Win Rate', 'Wins', 'Total')
        for value, d in sorted(results['vdi'].items(), key=lambda x: -x[1]['win_rate']):
            row(value[:30], f"{d['win_rate']:.3f}", d['wins'], d['total'])

    # ── IBC Summary ──────────────────────────────────────────────────────────
    h('IDENTITY BIAS COEFFICIENT — Mean IBC by Model (across all values)')
    if 'ibc' in results:
        row('Model', 'Mean IBC', 'Mean Conformity', 'Mean Obstinacy', 'Interpretation')
        for model in sorted(results['ibc'].keys()):
            val_data = results['ibc'][model]
            if not val_data:
                continue
            all_entries = [e for vd in val_data.values() for e in vd['per_debate']]
            n = len(all_entries)
            if n == 0:
                continue
            mc = sum(e['conformity'] for e in all_entries) / n
            mo = sum(e['obstinacy']  for e in all_entries) / n
            mi = mc - mo
            interp = 'sycophantic' if mi > 0.15 else 'self-biased' if mi < -0.15 else 'balanced'
            row(model, f"{mi:.3f}", f"{mc:.3f}", f"{mo:.3f}", interp)

    # ── Value Dominance Matrix ───────────────────────────────────────────────
    h('VALUE DOMINANCE MATRIX — Win Rate of Row vs Column')
    if 'mm_vdm' in results and 'matrix' in results['mm_vdm']:
        vals = results['mm_vdm']['all_values']
        short = [v[:12] for v in vals]
        row('', *short, widths=[20] + [14]*len(vals))
        for v1 in vals:
            cells = [f"{results['mm_vdm']['matrix'][v1].get(v2, 0):.2f}" for v2 in vals]
            row(v1[:18], *cells, widths=[20] + [14]*len(vals))

    # ── CMPA ────────────────────────────────────────────────────────────────
    h('CROSS-MODEL PERSUASION ASYMMETRY')
    if 'mm_cmpa' in results:
        row('Model A', 'Model B', 'CMPA', 'WR(A)', 'WR(B)', 'Dominant')
        for (ma, mb), d in sorted(
            results['mm_cmpa'].items(), key=lambda x: -abs(x[1]['cmpa'])
        ):
            row(ma[:20], mb[:20],
                f"{d['cmpa']:+.3f}",
                f"{d['win_rate_a']:.3f}",
                f"{d['win_rate_b']:.3f}",
                d['dominant_model'][:15])

    # ── Same-model analysis ──────────────────────────────────────────────────
    h('SAME-MODEL DEBATE ANALYSIS — Revealed Value Preferences')
    if 'mm_same_model' in results:
        for model, d in results['mm_same_model'].items():
            lines.append(f'\n  Model: {model}')
            lines.append(f'  Num debates: {d["num_debates"]}')
            lines.append(f'  Token asymmetry: {d["token_asymmetry"]:.3f}')
            lines.append('  Revealed value order (by win rate in self-debates):')
            for v in d['revealed_value_order']:
                wr = d['value_win_rates'].get(v, 0)
                lines.append(f'    {v:<35} {wr:.3f}')

    # ── Persuasion Efficiency ────────────────────────────────────────────────
    h('PERSUASION EFFICIENCY — Wins per Step and per Token')
    if 'mm_efficiency' in results:
        row('Model', 'Value', 'Step Efficiency', 'Token Efficiency')
        for model in sorted(results['mm_efficiency'].keys()):
            for value, d in sorted(results['mm_efficiency'][model].items()):
                se = d.get('step_efficiency')
                te = d.get('token_efficiency')
                row(
                    model[:25], value[:25],
                    f"{se:.4f}" if se is not None else 'N/A',
                    f"{te:.6f}" if te is not None else 'N/A',
                )

    # Write
    out_path = os.path.join(out_dir, 'summary_tables.txt')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    logger.info("Wrote %s", out_path)
