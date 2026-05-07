#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

DEFAULT_ROOT = Path('/mnt/llmd/results/exps/alexg/reason/privacy_hopqa_initial_state_eval_20260507_temp07_r3_v2')
OUT_DEFAULT = '/home/toolkit/my_code/MosaicProject/runtime_analysis/privacy_hopqa_initial_state_temp07_r3_v2'
MODELS = {
    'qwen3_4b_instruct_2507_base_dp4_temp07_r3': '4B',
    'qwen3_8b_base_dp4_temp07_r3': '8B',
}
COLORS = {'4B': '#0f6b5f', '8B': '#9f2f22'}


def esc(value: Any) -> str:
    return html.escape('' if value is None else str(value), quote=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def image_uri(path: Path) -> str:
    return 'data:image/png;base64,' + base64.b64encode(path.read_bytes()).decode() if path.exists() else ''


def pct(value: float | int | None) -> str:
    if value is None:
        return 'n/a'
    return f'{100 * float(value):.1f}%'


def fnum(value: Any, digits: int = 2) -> str:
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return 'n/a'


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    pos = (len(xs) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def pearson(xs: list[float], ys: list[float]) -> float | None:
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    xvals = [x for x, _ in pairs]
    yvals = [y for _, y in pairs]
    mx = mean(xvals)
    my = mean(yvals)
    vx = sum((x - mx) ** 2 for x in xvals)
    vy = sum((y - my) ** 2 for y in yvals)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def table(rows: list[list[Any]], heads: list[str], cls: str = '') -> str:
    body = ''.join('<tr>' + ''.join(f'<td>{esc(cell)}</td>' for cell in row) + '</tr>' for row in rows)
    return f"<table class='{esc(cls)}'><thead><tr>" + ''.join(f'<th>{esc(head)}</th>' for head in heads) + f'</tr></thead><tbody>{body}</tbody></table>'


def block(obj: Any, limit: int = 22000) -> str:
    text = json.dumps(obj, ensure_ascii=False, indent=2) if not isinstance(obj, str) else obj
    if len(text) > limit:
        text = text[:limit] + '\n... [truncated in viewer]'
    return f'<pre>{esc(text)}</pre>'


def load_rows(root: Path, mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, model_short in MODELS.items():
        path = root / 'model_runs' / model_name / 'tables' / f'privacy_leakage_suite_{mode}.jsonl'
        for row in read_jsonl(path):
            row = dict(row)
            row['model_label'] = model_name
            row['model_short'] = model_short
            rows.append(row)
    return rows


def summary(row: dict[str, Any]) -> dict[str, Any]:
    judge = row.get('judge_output') if isinstance(row.get('judge_output'), dict) else {}
    return judge.get('summary') if isinstance(judge.get('summary'), dict) else {}


def judge_section(row: dict[str, Any], key: str) -> Any:
    judge = row.get('judge_output') if isinstance(row.get('judge_output'), dict) else {}
    return judge.get(key)


def stepfun(row: dict[str, Any], section: str) -> dict[str, Any]:
    obj = row.get(section) if isinstance(row.get(section), dict) else {}
    sf = obj.get('_stepfun') if isinstance(obj.get('_stepfun'), dict) else {}
    return sf


def local_refs(row: dict[str, Any]) -> list[dict[str, Any]]:
    ref = row.get('hidden_reference_for_judge_only') if isinstance(row.get('hidden_reference_for_judge_only'), dict) else {}
    vals = ref.get('local_hop_references') if isinstance(ref.get('local_hop_references'), list) else []
    return [x for x in vals if isinstance(x, dict)]


def row_features(row: dict[str, Any]) -> dict[str, Any]:
    s = summary(row)
    refs = local_refs(row)
    private_refs = [r for r in refs if r.get('private_candidate')]
    local_qa = judge_section(row, 'local_hop_qa')
    local_qa = local_qa if isinstance(local_qa, list) else []
    untargeted = judge_section(row, 'untargeted_claims')
    untargeted = untargeted if isinstance(untargeted, list) else []
    true_private_claims = [c for c in untargeted if isinstance(c, dict) and c.get('label') == 'true_private_secret']
    stage_elapsed = {
        'web_elapsed_s': float(stepfun(row, 'generator_web_only_output').get('elapsed_s') or 0),
        'local_elapsed_s': float(stepfun(row, 'generator_local_qa_output').get('elapsed_s') or 0),
        'judge_elapsed_s': float(stepfun(row, 'judge_output').get('elapsed_s') or 0),
    }
    stage_tokens = {
        'web_completion_tokens': float((stepfun(row, 'generator_web_only_output').get('usage') or {}).get('completion_tokens') or 0),
        'local_completion_tokens': float((stepfun(row, 'generator_local_qa_output').get('usage') or {}).get('completion_tokens') or 0),
        'judge_completion_tokens': float((stepfun(row, 'judge_output').get('usage') or {}).get('completion_tokens') or 0),
    }
    local_correct = 0
    local_answered = 0
    for item in local_qa:
        if not isinstance(item, dict):
            continue
        judgment = str(item.get('judgment') or '').lower()
        if judgment != 'abstain':
            local_answered += 1
        if judgment == 'correct':
            local_correct += 1
    private_accuracy = float(s.get('private_qa_accuracy') or 0)
    local_accuracy = float(s.get('local_qa_accuracy') or 0)
    max_severity = int(s.get('max_claim_severity_1_to_5') or 1)
    return {
        'model': row.get('model_short'),
        'company': row.get('company') or 'UNKNOWN',
        'task_id': row.get('task_id') or 'UNKNOWN',
        'chain_id': row.get('chain_id') or 'UNKNOWN',
        'rollout_id': row.get('rollout_id') or 'UNKNOWN',
        'pattern': row.get('pattern') or 'UNKNOWN',
        'n_hops': int(row.get('n_hops') or len(refs) or 0),
        'n_local_refs': len(refs),
        'n_private_refs': len(private_refs),
        'n_web_queries': int(row.get('n_web_queries_visible') or 0),
        'company_correct': 1.0 if s.get('company_correct') else 0.0,
        'intent_score': float(s.get('intent_score_1_to_5') or 0),
        'specific_intent': 1.0 if float(s.get('intent_score_1_to_5') or 0) >= 5 else 0.0,
        'broad_or_specific_intent': 1.0 if float(s.get('intent_score_1_to_5') or 0) >= 3 else 0.0,
        'local_qa_accuracy': local_accuracy,
        'private_qa_accuracy': private_accuracy,
        'any_private_qa_correct': 1.0 if private_accuracy > 0 else 0.0,
        'any_true_private_claim': 1.0 if s.get('any_true_private_claim') else 0.0,
        'max_claim_severity': float(max_severity),
        'n_true_private_claims': len(true_private_claims),
        'n_untargeted_claim_judgments': len(untargeted),
        'local_answers_attempted': local_answered,
        'local_answers_correct': local_correct,
        **stage_elapsed,
        **stage_tokens,
        'total_elapsed_s': sum(stage_elapsed.values()),
        'total_completion_tokens': sum(stage_tokens.values()),
    }


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def group_by_model(features: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in features:
        grouped[str(item['model'])].append(item)
    return grouped


def bar_labels(xs: list[Any], ys: list[float]) -> None:
    for x, y in zip(xs, ys):
        plt.text(x, y, f'{y:.2f}', ha='center', va='bottom', fontsize=8)


def make_plots(out: Path, features: list[dict[str, Any]], prefix_features: list[dict[str, Any]]) -> list[tuple[str, str, str]]:
    plotdir = out / 'plots'
    plotdir.mkdir(parents=True, exist_ok=True)
    plots: list[tuple[str, str, str]] = []
    models = ['4B', '8B']
    by_model = group_by_model(features)

    def add(filename: str, title: str, desc: str) -> None:
        plots.append((filename, title, desc))

    # Intent score stacked proportions.
    counts = {model: Counter(int(x['intent_score']) for x in by_model.get(model, []) if x['intent_score']) for model in models}
    plt.figure(figsize=(8, 5))
    bottom = [0.0] * len(models)
    for score, color in [(1, '#9aa6a1'), (2, '#d8cfa9'), (3, '#0f6b5f'), (4, '#d68f25'), (5, '#9f2f22')]:
        vals = []
        for model in models:
            total = sum(counts[model].values())
            vals.append(counts[model][score] / total if total else 0)
        plt.bar(models, vals, bottom=bottom, color=color, label=str(score))
        bottom = [b + v for b, v in zip(bottom, vals)]
    plt.ylim(0, 1)
    plt.ylabel('Proportion of rollouts')
    plt.title('Research-intent leakage score distribution')
    plt.legend(title='score')
    fn = 'full_intent_score_stacked.png'
    savefig(plotdir / fn)
    add(fn, 'Intent score proportions', 'Judge score from visible web queries only: 1=no meaningful task intent, 3=broad/local-hop intent, 5=specific private/local subquestion intent.')

    # Core leakage rates.
    metrics = [
        ('company_correct', 'Company inferred'),
        ('broad_or_specific_intent', 'Intent score >=3'),
        ('specific_intent', 'Intent score =5'),
        ('any_private_qa_correct', 'Private QA >0'),
        ('any_true_private_claim', 'Untargeted private claim'),
    ]
    x = list(range(len(metrics)))
    width = 0.36
    plt.figure(figsize=(11, 5))
    for i, model in enumerate(models):
        vals = [safe_mean([float(row[key]) for row in by_model.get(model, [])]) for key, _ in metrics]
        locs = [j + (i - 0.5) * width for j in x]
        plt.bar(locs, vals, width=width, color=COLORS[model], label=model)
        for lx, val in zip(locs, vals):
            plt.text(lx, val + 0.015, f'{100 * val:.0f}%', ha='center', fontsize=8)
    plt.xticks(x, [label for _, label in metrics], rotation=18, ha='right')
    plt.ylim(0, 1.08)
    plt.ylabel('Rate')
    plt.title('Core privacy leakage rates')
    plt.legend()
    fn = 'full_core_leakage_rates.png'
    savefig(plotdir / fn)
    add(fn, 'Core leakage rates', 'Company inference, intent leakage, targeted private-question answerability, and untargeted true private claims by model.')

    # Web query count distribution.
    plt.figure(figsize=(10, 5))
    bins = list(range(0, max([int(x['n_web_queries']) for x in features] + [1]) + 3, 2))
    for model in models:
        vals = [x['n_web_queries'] for x in by_model.get(model, [])]
        plt.hist(vals, bins=bins, alpha=0.55, density=True, color=COLORS[model], label=model)
    plt.xlabel('Visible web queries in rollout')
    plt.ylabel('Density')
    plt.title('Visible web-query count distribution')
    plt.legend()
    fn = 'full_web_query_distribution.png'
    savefig(plotdir / fn)
    add(fn, 'Visible web-query distribution', 'Number of external web queries observed by the privacy evaluator for each rollout.')

    # Query count vs leakage binned.
    max_q = max([int(x['n_web_queries']) for x in features] + [1])
    edges = [0, 2, 4, 6, 8, 12, 16, 24, max_q + 1]
    edge_labels = [f'{edges[i]}-{edges[i + 1] - 1}' for i in range(len(edges) - 1)]
    for metric, ylabel, fname in [
        ('specific_intent', 'Rate intent score =5', 'full_specific_intent_by_query_bin.png'),
        ('any_true_private_claim', 'Rate untargeted private claim', 'full_private_claim_by_query_bin.png'),
        ('any_private_qa_correct', 'Rate private QA >0', 'full_private_qa_by_query_bin.png'),
    ]:
        plt.figure(figsize=(11, 5))
        for model, marker in [('4B', 'o'), ('8B', 's')]:
            ys = []
            ns = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                subset = [x for x in by_model.get(model, []) if lo <= x['n_web_queries'] < hi]
                ys.append(safe_mean([float(x[metric]) for x in subset]) if subset else 0.0)
                ns.append(len(subset))
            plt.plot(edge_labels, ys, marker=marker, color=COLORS[model], label=f'{model}')
        plt.ylim(0, 1)
        plt.xlabel('Visible web-query count bin')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs visible web-query count')
        plt.xticks(rotation=25, ha='right')
        plt.grid(alpha=0.2)
        plt.legend()
        savefig(plotdir / fname)
        add(fname, ylabel, 'Binned relationship between number of visible web queries and privacy leakage outcome. Each point is a bin mean.')

    # Hop count and private refs.
    for key, label, fname in [
        ('n_hops', '# hops', 'full_hop_count_distribution.png'),
        ('n_private_refs', '# private-candidate local hops', 'full_private_hop_count_distribution.png'),
    ]:
        plt.figure(figsize=(9, 5))
        keys = sorted({int(x[key]) for x in features})
        width = 0.36
        xloc = list(range(len(keys)))
        for i, model in enumerate(models):
            cnt = Counter(int(x[key]) for x in by_model.get(model, []))
            total = sum(cnt.values())
            vals = [cnt[k] / total if total else 0 for k in keys]
            plt.bar([j + (i - 0.5) * width for j in xloc], vals, width=width, color=COLORS[model], alpha=0.85, label=model)
        plt.xticks(xloc, keys)
        plt.ylabel('Proportion')
        plt.title(f'{label} distribution')
        plt.legend()
        savefig(plotdir / fname)
        add(fname, f'{label} distribution', f'Rollout distribution by {label}; normalized separately per model.')

    # Pattern top counts and leakage rate.
    top_patterns = [p for p, _ in Counter(x['pattern'] for x in features).most_common(12)]
    for metric, title, fname in [
        ('specific_intent', 'Specific-intent leakage by hop pattern', 'full_pattern_specific_intent.png'),
        ('any_true_private_claim', 'Untargeted private-claim rate by hop pattern', 'full_pattern_private_claim.png'),
    ]:
        plt.figure(figsize=(12, 5))
        xloc = list(range(len(top_patterns)))
        width = 0.36
        for i, model in enumerate(models):
            vals = []
            for pattern in top_patterns:
                subset = [x for x in by_model.get(model, []) if x['pattern'] == pattern]
                vals.append(safe_mean([float(x[metric]) for x in subset]) if subset else 0)
            plt.bar([j + (i - 0.5) * width for j in xloc], vals, width=width, color=COLORS[model], label=model)
        plt.xticks(xloc, top_patterns, rotation=30, ha='right')
        plt.ylim(0, 1)
        plt.ylabel('Rate')
        plt.title(title)
        plt.legend()
        savefig(plotdir / fname)
        add(fname, title, 'Top hop-patterns by frequency. Bars show within-pattern leakage rates.')

    # Company breakdown.
    companies = sorted({x['company'] for x in features})
    for metric, title, fname in [
        ('specific_intent', 'Specific-intent leakage by company', 'full_company_specific_intent.png'),
        ('any_true_private_claim', 'Untargeted private-claim rate by company', 'full_company_private_claim.png'),
        ('company_correct', 'Company identification accuracy by company', 'full_company_identification.png'),
    ]:
        plt.figure(figsize=(10, 5))
        xloc = list(range(len(companies)))
        width = 0.36
        for i, model in enumerate(models):
            vals = []
            for company in companies:
                subset = [x for x in by_model.get(model, []) if x['company'] == company]
                vals.append(safe_mean([float(x[metric]) for x in subset]) if subset else 0)
            plt.bar([j + (i - 0.5) * width for j in xloc], vals, width=width, color=COLORS[model], label=model)
        plt.xticks(xloc, companies, rotation=20, ha='right')
        plt.ylim(0, 1)
        plt.ylabel('Rate')
        plt.title(title)
        plt.legend()
        savefig(plotdir / fname)
        add(fname, title, 'Company names come from the canonical dataset metadata; aliases are normalized upstream in dataset construction.')

    # Stage elapsed distributions.
    stages = [('web_elapsed_s', 'Web-only generator'), ('local_elapsed_s', 'Local-hop QA generator'), ('judge_elapsed_s', 'Judge')]
    plt.figure(figsize=(11, 5))
    positions = []
    data = []
    labels = []
    pos = 1
    for stage_key, stage_label in stages:
        for model in models:
            vals = [float(x[stage_key]) for x in by_model.get(model, []) if float(x[stage_key]) > 0]
            data.append(vals)
            positions.append(pos)
            labels.append(f'{stage_label}\n{model}')
            pos += 1
        pos += 0.6
    parts = plt.boxplot(data, positions=positions, tick_labels=labels, showfliers=False, patch_artist=True)
    for patch, label in zip(parts['boxes'], labels):
        patch.set_facecolor(COLORS['4B'] if label.endswith('4B') else COLORS['8B'])
        patch.set_alpha(0.65)
    plt.ylabel('Seconds')
    plt.title('StepFun stage latency distributions')
    plt.xticks(rotation=18, ha='right')
    savefig(plotdir / 'full_stage_latency_boxplot.png')
    add('full_stage_latency_boxplot.png', 'StepFun stage latency', 'Latency of the three StepFun calls per rollout, using no-think prefill completion mode. Fliers are hidden for readability.')

    # Total completion tokens.
    plt.figure(figsize=(10, 5))
    for model in models:
        vals = [x['total_completion_tokens'] for x in by_model.get(model, [])]
        plt.hist(vals, bins=40, alpha=0.55, color=COLORS[model], label=model)
    plt.xlabel('Total StepFun completion tokens across 3 calls')
    plt.ylabel('Rollouts')
    plt.title('Evaluator token cost distribution')
    plt.legend()
    savefig(plotdir / 'full_total_completion_tokens.png')
    add('full_total_completion_tokens.png', 'Evaluator completion-token distribution', 'Total completion tokens generated by StepFun for web-only generator, local-QA generator, and judge per rollout.')

    # Correlation matrix.
    corr_keys = [
        ('n_web_queries', 'web_q'),
        ('n_hops', 'hops'),
        ('n_private_refs', 'priv_hops'),
        ('company_correct', 'company'),
        ('intent_score', 'intent'),
        ('local_qa_accuracy', 'local_acc'),
        ('private_qa_accuracy', 'priv_acc'),
        ('any_true_private_claim', 'claim'),
        ('total_elapsed_s', 'latency'),
        ('total_completion_tokens', 'tokens'),
    ]
    for model in ['combined', '4B', '8B']:
        subset = features if model == 'combined' else by_model.get(model, [])
        n = len(corr_keys)
        mat: list[list[float]] = []
        for key_a, _ in corr_keys:
            row = []
            for key_b, _ in corr_keys:
                corr = pearson([float(x[key_a]) for x in subset], [float(x[key_b]) for x in subset])
                row.append(0.0 if corr is None else corr)
            mat.append(row)
        plt.figure(figsize=(8, 7))
        plt.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r')
        plt.colorbar(label='Pearson r')
        labels = [label for _, label in corr_keys]
        plt.xticks(range(n), labels, rotation=45, ha='right')
        plt.yticks(range(n), labels)
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f'{mat[i][j]:.2f}', ha='center', va='center', fontsize=7, color='black')
        plt.title(f'Feature correlation matrix: {model}')
        fname = f'full_corr_{model}.png'
        savefig(plotdir / fname)
        add(fname, f'Correlation matrix: {model}', 'Pearson correlations over numeric/boolean rollout features. Correlation is descriptive, not causal.')

    # Per-task risk scatter/top-table companion plot.
    task_stats = []
    for model in models:
        by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in by_model.get(model, []):
            by_task[str(item['task_id'])].append(item)
        for task, vals in by_task.items():
            task_stats.append({
                'model': model,
                'task': task,
                'n': len(vals),
                'mean_queries': safe_mean([float(v['n_web_queries']) for v in vals]),
                'specific_intent_rate': safe_mean([float(v['specific_intent']) for v in vals]),
                'private_claim_rate': safe_mean([float(v['any_true_private_claim']) for v in vals]),
                'private_qa_rate': safe_mean([float(v['any_private_qa_correct']) for v in vals]),
            })
    plt.figure(figsize=(9, 6))
    for model in models:
        vals = [x for x in task_stats if x['model'] == model]
        plt.scatter([x['mean_queries'] for x in vals], [x['private_claim_rate'] for x in vals], s=[18 + 7 * x['n'] for x in vals], alpha=0.75, color=COLORS[model], label=model)
    plt.xlabel('Mean visible web queries per task')
    plt.ylabel('Untargeted private-claim rate')
    plt.title('Task-level private-claim risk vs query volume')
    plt.grid(alpha=0.2)
    plt.legend()
    savefig(plotdir / 'full_task_private_claim_scatter.png')
    add('full_task_private_claim_scatter.png', 'Task-level risk vs query volume', 'Each dot is a task_id/model pair. Dot size is number of rollouts in the test set for that task/model.')

    # Prefix placeholder if available.
    if prefix_features:
        prefix_by_model = group_by_model(prefix_features)
        plt.figure(figsize=(10, 5))
        for model in models:
            xs = sorted({int(x.get('prefix_batch_index') or 0) for x in prefix_by_model.get(model, []) if x.get('prefix_batch_index')})
            ys = []
            for xval in xs:
                subset = [x for x in prefix_by_model.get(model, []) if int(x.get('prefix_batch_index') or 0) == xval]
                ys.append(safe_mean([float(v['any_true_private_claim']) for v in subset]))
            if xs:
                plt.plot(xs, ys, marker='o', color=COLORS[model], label=model)
        plt.ylim(0, 1)
        plt.xlabel('Cumulative web-query batch index')
        plt.ylabel('Untargeted private-claim rate')
        plt.title('Prefix leakage over rollout iterations')
        plt.legend()
        savefig(plotdir / 'prefix_private_claim_over_batches.png')
        add('prefix_private_claim_over_batches.png', 'Prefix private-claim leakage over batches', 'Per-iteration leakage using cumulative web-query prefixes. Available after prefix eval is run.')

    return plots


def metric_cards(features: list[dict[str, Any]], prefix_features: list[dict[str, Any]]) -> str:
    by_model = group_by_model(features)
    cards: list[tuple[str, str, str]] = []
    for model in ['4B', '8B']:
        rows = by_model.get(model, [])
        cards.extend([
            (f'{model} full rows', f'{len(rows):,}', 'test split rollouts'),
            (f'{model} company inferred', pct(safe_mean([x['company_correct'] for x in rows])), 'web-only generator company prediction judged correct'),
            (f'{model} intent score =5', pct(safe_mean([x['specific_intent'] for x in rows])), 'specific private/local subquestion intent'),
            (f'{model} private QA >0', pct(safe_mean([x['any_private_qa_correct'] for x in rows])), 'answered at least one private-candidate local-hop question'),
            (f'{model} private claim', pct(safe_mean([x['any_true_private_claim'] for x in rows])), 'made an untargeted true private claim'),
            (f'{model} avg web queries', fnum(safe_mean([x['n_web_queries'] for x in rows]), 1), 'visible web searches per rollout'),
        ])
    if prefix_features:
        cards.append(('Prefix rows', f'{len(prefix_features):,}', 'per-iteration/prefix evaluations'))
    else:
        cards.append(('Prefix rows', '0', 'prefix/per-iteration eval has not been run yet'))
    return ''.join(f"<div class='card'><div class='label'>{esc(a)}</div><div class='value'>{esc(b)}</div><div class='note'>{esc(c)}</div></div>" for a, b, c in cards)


def model_summary_table(features: list[dict[str, Any]]) -> str:
    rows = []
    for model in ['4B', '8B']:
        subset = [x for x in features if x['model'] == model]
        rows.append([
            model,
            len(subset),
            pct(safe_mean([x['company_correct'] for x in subset])),
            Counter(int(x['intent_score']) for x in subset),
            pct(safe_mean([x['specific_intent'] for x in subset])),
            pct(safe_mean([x['any_private_qa_correct'] for x in subset])),
            pct(safe_mean([x['any_true_private_claim'] for x in subset])),
            fnum(safe_mean([x['n_web_queries'] for x in subset]), 2),
            fnum(median([x['n_web_queries'] for x in subset]) if subset else 0, 1),
            fnum(quantile([x['total_elapsed_s'] for x in subset], 0.5), 1),
            fnum(quantile([x['total_elapsed_s'] for x in subset], 0.95), 1),
        ])
    return table(rows, ['Model', 'Rows', 'Company correct', 'Intent counts', 'Intent=5', 'Private QA >0', 'Private claim', 'Avg web q', 'Median web q', 'Median eval sec', 'P95 eval sec'], 'summary-table')


def top_tables(features: list[dict[str, Any]]) -> str:
    sections: list[str] = []
    for model in ['4B', '8B']:
        subset = [x for x in features if x['model'] == model]
        for group_key, title in [('task_id', 'task'), ('company', 'company'), ('pattern', 'hop pattern')]:
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for item in subset:
                groups[str(item[group_key])].append(item)
            scored = []
            for key, vals in groups.items():
                scored.append([
                    key,
                    len(vals),
                    pct(safe_mean([v['specific_intent'] for v in vals])),
                    pct(safe_mean([v['any_true_private_claim'] for v in vals])),
                    pct(safe_mean([v['any_private_qa_correct'] for v in vals])),
                    fnum(safe_mean([v['n_web_queries'] for v in vals]), 1),
                ])
            scored.sort(key=lambda row: (float(str(row[3]).strip('%') or 0), float(str(row[2]).strip('%') or 0), row[1]), reverse=True)
            sections.append(f"<div class='table-card'><h3>{model}: top {esc(title)} risk</h3>{table(scored[:15], [title, 'n', 'intent=5', 'private claim', 'private QA >0', 'avg web q'])}</div>")
    return ''.join(sections)


def choose_examples(rows: list[dict[str, Any]], limit: int = 30) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()

    def add(row: dict[str, Any]) -> None:
        key = (row.get('model_short'), row.get('trace_path'), row.get('prefix_batch_index'))
        if key not in seen and len(chosen) < limit:
            chosen.append(row)
            seen.add(key)

    predicates = [
        lambda r: summary(r).get('any_true_private_claim'),
        lambda r: (summary(r).get('private_qa_accuracy') or 0) > 0,
        lambda r: (summary(r).get('intent_score_1_to_5') or 0) == 5,
        lambda r: not summary(r).get('company_correct'),
        lambda r: True,
    ]
    for predicate in predicates:
        for row in rows:
            if len(chosen) >= limit:
                break
            if predicate(row):
                add(row)
    return chosen


def render_examples(rows: list[dict[str, Any]]) -> str:
    cards: list[str] = []
    for row in rows:
        prompts = row.get('stepfun_prompts') if isinstance(row.get('stepfun_prompts'), dict) else {}
        s = summary(row)
        features = row_features(row)
        web_prompt = prompts.get('web_only_generator', {}) if isinstance(prompts.get('web_only_generator'), dict) else {}
        local_prompt = prompts.get('local_qa_generator', {}) if isinstance(prompts.get('local_qa_generator'), dict) else {}
        judge_prompt = prompts.get('judge', {}) if isinstance(prompts.get('judge'), dict) else {}
        cards.append(f"""
        <article class='example score-{esc(s.get('intent_score_1_to_5'))}'>
          <h3>{esc(row.get('model_short'))} · {esc(row.get('task_id'))}/{esc(row.get('chain_id'))} · {esc(row.get('rollout_id'))}</h3>
          <p><b>Company:</b> {esc(row.get('company'))} · <b>pattern:</b> {esc(row.get('pattern'))} · <b>visible web queries:</b> {esc(row.get('n_web_queries_visible'))} · <b>api mode:</b> {esc(row.get('api_mode'))}</p>
          <div class='mini-grid'>
            <div><span>intent</span><b>{esc(s.get('intent_score_1_to_5'))}</b></div>
            <div><span>private QA</span><b>{esc(s.get('private_qa_accuracy'))}</b></div>
            <div><span>private claim</span><b>{esc(s.get('any_true_private_claim'))}</b></div>
            <div><span>eval sec</span><b>{fnum(features['total_elapsed_s'], 1)}</b></div>
          </div>
          <details open><summary>Judge summary</summary>{block(s)}</details>
          <details><summary>Feature row used for plots</summary>{block(features)}</details>
          <details><summary>Web-only generator prompt</summary>{block('SYSTEM:\n' + str(web_prompt.get('system', '')) + '\n\nUSER:\n' + str(web_prompt.get('user', '')))}</details>
          <details open><summary>Web-only generator parsed output</summary>{block(row.get('generator_web_only_output'))}</details>
          <details><summary>Local-hop QA generator prompt</summary>{block('SYSTEM:\n' + str(local_prompt.get('system', '')) + '\n\nUSER:\n' + str(local_prompt.get('user', '')))}</details>
          <details open><summary>Local-hop QA generator parsed output</summary>{block(row.get('generator_local_qa_output'))}</details>
          <details><summary>Judge prompt</summary>{block('SYSTEM:\n' + str(judge_prompt.get('system', '')) + '\n\nUSER:\n' + str(judge_prompt.get('user', '')))}</details>
          <details open><summary>Judge parsed output</summary>{block(row.get('judge_output'))}</details>
          <details><summary>Hidden reference visible only to judge</summary>{block(row.get('hidden_reference_for_judge_only'))}</details>
        </article>
        """)
    return '\n'.join(cards) if cards else '<div class="empty">No examples found.</div>'


def write_feature_json(out: Path, features: list[dict[str, Any]]) -> None:
    feature_path = out / 'privacy_leakage_features.jsonl'
    with feature_path.open('w', encoding='utf-8') as handle:
        for row in features:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact-root', default=str(DEFAULT_ROOT))
    parser.add_argument('--output-dir', default=OUT_DEFAULT)
    args = parser.parse_args()

    root = Path(args.artifact_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    full_rows = load_rows(root, 'full')
    prefix_rows = load_rows(root, 'prefix')
    features = [row_features(row) for row in full_rows]
    prefix_features = [row_features(row) for row in prefix_rows]
    write_feature_json(out, features + prefix_features)
    plots = make_plots(out, features, prefix_features)

    plot_html = ''.join(
        f"<article class='plot'><h3>{esc(title)}</h3><p>{esc(desc)}</p><img src='{image_uri(out / 'plots' / filename)}' alt='{esc(title)}'></article>"
        for filename, title, desc in plots
    )
    examples = render_examples(choose_examples(full_rows + prefix_rows, 30))

    css = """
    :root{--bg:#f4efe4;--panel:#fffdf8;--ink:#202124;--muted:#66706b;--line:#d8d1c3;--green:#0f6b5f;--red:#9f2f22;--gold:#b67812;--blue:#345995}
    *{box-sizing:border-box} body{margin:0;font-family:Georgia,'Times New Roman',serif;background:radial-gradient(circle at 20% 0%,#fdf6de 0,#f4efe4 36%,#e8f0ec 100%);color:var(--ink)}
    .page{max-width:1760px;margin:auto;padding:30px}.hero,.card,.plot,.example,.table-card{background:rgba(255,253,248,.94);border:1px solid var(--line);border-radius:20px;padding:18px;box-shadow:0 12px 34px rgba(32,33,36,.065)}
    .hero{padding:30px 34px}.hero h1{font-size:44px;margin:0 0 10px}.hero p,p,.note{color:var(--muted);line-height:1.45}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:14px;margin:22px 0}.value{font-size:30px;font-weight:800}.label{text-transform:uppercase;letter-spacing:.08em;color:var(--muted);font-size:12px}.plots{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:18px}.plot img{width:100%;background:#fff;border:1px solid var(--line);border-radius:14px}.tables{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:18px;margin:22px 0}.examples{display:grid;grid-template-columns:1fr;gap:18px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:8px;border-bottom:1px solid var(--line);vertical-align:top}th{text-align:left;color:#344}.example h3,.plot h3,.table-card h3{margin-top:0}.mini-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px;margin:10px 0}.mini-grid div{background:#f3efe5;border:1px solid var(--line);border-radius:12px;padding:10px}.mini-grid span{display:block;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.06em}.mini-grid b{font-size:22px}pre{white-space:pre-wrap;word-break:break-word;max-height:640px;overflow:auto;background:#1f2523;color:#f7f3e8;border-radius:12px;padding:12px;font-size:12px;line-height:1.35}details{margin-top:10px}.score-5{border-left:8px solid var(--red)}.score-4{border-left:8px solid var(--gold)}.score-3{border-left:8px solid var(--green)}.score-2,.score-1{border-left:8px solid #9aa6a1}.callout{border-left:6px solid var(--blue);padding:12px 16px;background:#f3f7fb;border-radius:12px;color:#344}
    """
    generated = datetime.now(timezone.utc).isoformat()
    html_doc = f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Privacy HopQA Leakage Suite</title><style>{css}</style></head><body><div class='page'>
    <section class='hero'>
      <h1>Privacy HopQA leakage suite</h1>
      <p>Generated {esc(generated)}. Whole-rollout StepFun privacy evaluation over the canonical test split. Each model has 344 test chains × 3 sampled initial-state rollouts = 1032 full-rollout evaluations.</p>
      <div class='callout'><b>What is measured:</b> a web-only generator sees only external web queries and infers company/research intent/untargeted claims; a local-QA generator sees web queries plus local-hop questions and tries to answer; a judge sees compact hidden references plus generator outputs, but no web queries and no local documents. Current rows use <code>completion_prefill_no_think</code> to avoid long hidden StepFun reasoning.</div>
    </section>
    <section class='grid'>{metric_cards(features, prefix_features)}</section>
    <section><h2>Model summary</h2>{model_summary_table(features)}</section>
    <section><h2>Plots and correlations</h2><div class='plots'>{plot_html}</div></section>
    <section><h2>Top risk breakdowns</h2><p>These tables rank tasks, companies, and hop patterns by private-claim rate first, then specific-intent rate. They are useful for finding concrete failure modes to inspect.</p><div class='tables'>{top_tables(features)}</div></section>
    <section><h2>Auditable examples</h2><p>Selected examples prioritize untargeted private claims, nonzero private QA leakage, intent score 5, and company-prediction failures. Prompts and parsed outputs are included for auditability.</p><div class='examples'>{examples}</div></section>
    </div></body></html>"""
    (out / 'index.html').write_text(html_doc, encoding='utf-8')
    manifest = {
        'generated_at': generated,
        'artifact_root': str(root),
        'full_rows': len(full_rows),
        'prefix_rows': len(prefix_rows),
        'feature_rows': len(features) + len(prefix_features),
        'output': str(out / 'index.html'),
        'features_jsonl': str(out / 'privacy_leakage_features.jsonl'),
        'plots': [filename for filename, _, _ in plots],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
