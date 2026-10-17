"""Interactive HSTU trajectory simulator (Dash), served on this instance only (data stays in SPC AWS).
Pick a random5 TEST landmark (scored by the fold model that never saw the patient), edit its visible history (stage, genomic
findings, TMB, regimen at the landmark) and compare the predicted survival curve with the unedited one. Edits are applied to the
model input only: they show what the model learned (associations), not causal treatment effects.
Open via the JupyterLab proxy: <your JupyterLab URL with /lab replaced by>/proxy/8050/
usage: python dashboard.py [--port 8050]"""
import argparse, os, numpy as np, pandas as pd, torch
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update
from data import Events, ETYPES, truncate
from simulate import load_fold, token_id, arrays, add_token, insert_event, set_tokens, survival

p = argparse.ArgumentParser(); p.add_argument('--port', type=int, default=8050); a = p.parse_args()
torch.set_num_threads(1)                                  # CPU inference; the GPU and other cores belong to the training queues
DEVICE = 'cpu'

# ---------------- data (loaded once)
E = Events(); lm = E.lm
fold_of = {pid: f['name'] for f in E.sp['random5'] for pid in f['test']}
lm = lm.assign(fold=lm.patient_id.map(fold_of))
pat_no = {pid: i + 1 for i, pid in enumerate(sorted(lm.patient_id.unique()))}     # display number instead of record IDs
GENOMIC_FIND = {'MUT', 'DEL', 'AMP', 'FUSION'}
tok_str, etype_str = E.tokens, E.ev.etype.values
prefix_tokens = {l: set(tok_str[E.P[l]]) for l in lm.landmark_id.values}
lm = lm.assign(stk_keap=[bool({'MUT:STK11', 'MUT:KEAP1'} & prefix_tokens[l]) for l in lm.landmark_id.values],
               immuno=(lm.dclass.values & 2) > 0, platinum=lm.reg_name.str.contains('platin').values)
STAGES = sorted(t.split('=', 1)[1] for t in set(tok_str) if t.startswith('STATIC:STAGE='))
TMB_BUCKETS = ['0-0.5', '0.5-1.5', '1.5-3', '3-6', '>6']
mut_freq = pd.Series(tok_str[etype_str == 'MUT']).value_counts()
GENES = [t.split(':', 1)[1] for t in mut_freq[mut_freq >= 20].index]
reg_dclass = lm.groupby('reg_name').dclass.agg(lambda s: int(s.mode().iloc[0]))
reg_by_cohort = {c: d.reg_name.value_counts().index[:40].tolist() for c, d in lm.groupby('cohort')}
SCAN_VALS, MD_VALS = ['progressing', 'stable', 'improving', 'mixed'], ['progressing', 'stable', 'improving', 'mixed']
MET_SITES = pd.Series(tok_str[etype_str == 'MET']).value_counts().index.str.split(':').str[1].tolist()
FOLDS = {}


def fold_model(fold):
    if fold not in FOLDS: FOLDS[fold] = load_fold(E, fold, device=DEVICE)
    return FOLDS[fold]


LANES = [('Treatment', {'REG_START', 'REG_END'}), ('Oncologist notes', {'MD'}), ('Imaging', {'SCAN'}), ('Metastases', {'MET'}),
         ('Tumor markers', {'TM'}), ('Pathology', {'PATH'}), ('Genomics', {'PANEL', 'SAMPLE', 'TMB', 'MUT', 'FUSION', 'AMP', 'DEL'})]
LANE_OF = {e: name for name, es in LANES for e in es}
C = dict(bg='#0e1117', card='#161b26', line='#262d3d', text='#e6e9ef', muted='#8a93a6', accent='#7aa2ff', cf='#ff8a5c',
         base='#9aa3b2', good='#3ecf8e', bad='#ff6b6b')
LANE_COLOR = {'Treatment': '#7aa2ff', 'Oncologist notes': '#b38cff', 'Imaging': '#4fd1c5', 'Metastases': '#ff6b6b',
              'Tumor markers': '#f6c35b', 'Pathology': '#9aa3b2', 'Genomics': '#3ecf8e'}


def layout_fig(fig, h, **kw):
    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=h,
                      margin=dict(l=10, r=10, t=30, b=10), font=dict(color=C['text'], size=12),
                      legend=dict(orientation='h', y=1.08, x=0, bgcolor='rgba(0,0,0,0)'), **kw)
    fig.update_xaxes(gridcolor=C['line'], zerolinecolor=C['line']); fig.update_yaxes(gridcolor=C['line'], zerolinecolor=C['line'])
    return fig


# ---------------- per-landmark helpers
def landmark(lid):
    r = lm[lm.landmark_id == lid].iloc[0]
    Fd, model = fold_model(r.fold)
    ix = truncate(E, E.P[lid])
    return r, Fd, model, ix


def findings(ix):
    """(position in prefix, token) of genomic findings that can be removed"""
    return [(k, tok_str[j]) for k, j in enumerate(ix) if etype_str[j] in GENOMIC_FIND]


def edited(r, Fd, ix, stage, regimen, add_genes, remove_pos, tmb, scan=None, md=None, mets=None):
    s = arrays(E, Fd, ix)
    if remove_pos:                                         # by position, before anything shifts
        keep = np.ones(len(s['tok']), bool); keep[list(remove_pos)] = False
        s = {k: v[keep] for k, v in s.items()}
    if stage:
        stage_ids = [token_id(E, Fd, f'STATIC:STAGE={x}') for x in STAGES]
        s = {k: v.copy() for k, v in s.items()}
        s['tok'][(s['static'] == 1) & np.isin(s['tok'], stage_ids)] = token_id(E, Fd, f'STATIC:STAGE={stage}')
    if tmb and (s['etype'] == ETYPES.index('TMB')).any():
        s = set_tokens(s, ETYPES.index('TMB'), token_id(E, Fd, f'TMB:{tmb}'))
    if regimen and regimen != r.reg_name:
        s = {k: v.copy() for k, v in s.items()}
        s['tok'][-1] = token_id(E, Fd, f'REG_START:{regimen}'); s['dclass'][-1] = reg_dclass[regimen]
    present = {tok_str[ix[k]] for k in range(len(ix)) if k not in set(remove_pos or [])}
    for g in add_genes or []:
        if f'MUT:{g}' not in present:
            s = add_token(s, ETYPES.index('MUT'), token_id(E, Fd, f'MUT:{g}'))
    # disease status at this point: events dated on the landmark day, just before the regimen start
    for et, val in [('MET', m) for m in mets or []] + [('MD', md), ('SCAN', scan)]:
        if val:
            s = insert_event(s, ETYPES.index(et), token_id(E, Fd, f'{et}:{val}'), r.landmark_day)
    return s


def stage_options(Fd):
    """stages the fold model has its own embedding for; rare ones (pretrain count < 5 -> UNK) are shown disabled"""
    ok = lambda x: (token_id(E, Fd, f'STATIC:STAGE={x}') or 0) >= len(ETYPES)
    return [{'label': x if ok(x) else f'{x} (too rare for the model)', 'value': x, 'disabled': not ok(x)} for x in STAGES]


def median_months(S):
    k = np.where(S < 0.5)[0]
    return f'{k[0] + 1} mo' if len(k) else '>60 mo'


# ---------------- UI pieces
def card(children, **style):
    return html.Div(children, className='card', style=style)


def kpi(label, base, cf, fmt, lower_is_better=True):
    d = cf - base if isinstance(base, (int, float)) else None
    col = C['muted'] if d is None or abs(d) < 0.005 else (C['good'] if (d < 0) == lower_is_better else C['bad'])
    delta = '' if d is None else f'{"+" if d >= 0 else ""}{100 * d:.1f} pp'
    return html.Div([html.Div(label, className='kpi-label'),
                     html.Div([html.Span(fmt(base), className='kpi-base'), html.Span(' → ', className='kpi-arrow'),
                               html.Span(fmt(cf), className='kpi-value')]),
                     html.Div(delta, className='kpi-delta', style={'color': col})], className='kpi')


def dd(id_, label, **kw):
    return html.Div([html.Label(label, className='ctl-label'), dcc.Dropdown(id=id_, className='dd', **kw)], className='ctl')


prefix = os.environ.get('DASH_PREFIX', f'/jupyterlab/default/proxy/{a.port}/')
app = Dash(__name__, requests_pathname_prefix=prefix, routes_pathname_prefix='/', title='HSTU simulator',
           assets_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dashboard_assets'))

sim_tab = html.Div([
    html.Div([
        dd('cohort', 'Cohort', options=sorted(lm.cohort.unique()), value='NSCLC', clearable=False),
        dd('filter', 'Show', options=[{'label': 'All test landmarks', 'value': 'all'},
                                      {'label': 'STK11 / KEAP1 mutant', 'value': 'stk'},
                                      {'label': 'On immunotherapy', 'value': 'io'},
                                      {'label': 'On platinum', 'value': 'plat'}], value='all', clearable=False),
        html.Div(dd('patient', 'Patient', clearable=False), style={'flex': '2'}),
        html.Button('🎲 Random', id='random', className='btn'),
    ], className='row'),
    card([html.Div('Prognosis at… (each regimen start is a point where the model can predict, using everything up to that day)',
                   className='card-title'),
          dcc.Slider(id='step', min=0, max=0, step=None, value=0, marks={}, className='slider'),
          dcc.Graph(id='trajectory', config={'displayModeBar': False})]),
    dcc.Store(id='landmark'),
    html.Div(id='patient-card', className='row'),
    card([html.Div('Visible history up to this point (what the model sees)', className='card-title'),
          dcc.Graph(id='timeline', config={'displayModeBar': False})]),
    html.Div([
        card([html.Div('What if… at this point', className='card-title'),
              html.Div('Disease status now', className='sec'),
              dd('mets', 'New metastasis found now', options=MET_SITES, multi=True, placeholder='e.g. liver, brain'),
              dd('scan', 'Latest imaging result', options=SCAN_VALS, placeholder='as recorded'),
              dd('md', 'Oncologist assessment', options=MD_VALS, placeholder='as recorded'),
              dd('regimen', 'Regimen starting now', clearable=False),
              html.Div('Tumor genomics', className='sec'),
              dd('add', 'Add mutation', options=GENES, multi=True, placeholder='e.g. STK11, KEAP1, EGFR'),
              html.Div([html.Label('Remove genomic findings', className='ctl-label'),
                        dcc.Checklist(id='remove', className='chk', inputClassName='chk-in')], className='ctl'),
              dd('tmb', 'Tumor mutational burden (mut / 100 panel genes)', options=TMB_BUCKETS, clearable=False),
              html.Div('At diagnosis', className='sec'),
              dd('stage', 'Stage at diagnosis (recorded once; later spread shows up as metastases / scans above)', clearable=False),
              html.Button('Reset edits', id='reset', className='btn btn-ghost')], flex='1', minWidth='280px'),
        card([html.Div(id='kpis', className='kpis'),
              dcc.Graph(id='survival', config={'displayModeBar': False}),
              html.Div('Edits change only the model input, one token at a time. Differences show what the model learned '
                       '(associations in this cohort), not the effect of actually changing a patient\'s treatment or tumor.',
                       className='note')], flex='2', minWidth='320px'),
    ], className='row', style={'alignItems': 'stretch'}),
])


def checks_tab():
    if not os.path.exists('biology_results.csv'):
        return html.Div('Run biology.py first.', className='note')
    b = pd.read_csv('biology_results.csv'); fl = b[b.part == 'flip'].copy(); ob = b[b.part == 'observed']
    fl['label'] = fl.check + ' · ' + fl.group
    fl = fl.iloc[::-1]
    fig = go.Figure(go.Scatter(x=fl.delta_pp, y=fl.label, mode='markers', marker=dict(size=10, color=C['accent']),
                               error_x=dict(type='data', symmetric=False, array=fl.ci_hi - fl.delta_pp,
                                            arrayminus=fl.delta_pp - fl.ci_lo, color=C['accent']),
                               hovertemplate='%{y}<br>%{x:.2f} pp<extra></extra>'))
    fig.add_vline(x=0, line_color=C['muted'], line_dash='dot')
    layout_fig(fig, 60 + 34 * len(fl), xaxis_title='Change in predicted 12-month risk (percentage points), 95% CI')
    hr = go.Figure()
    for tag, col, nm in (('uni', C['base'], 'univariate'), ('adj', C['cf'], 'adjusted for line + stage')):
        hr.add_trace(go.Scatter(x=ob[f'HR_{tag}'], y=ob.check, mode='markers', name=nm, marker=dict(size=10, color=col),
                                error_x=dict(type='data', symmetric=False, array=ob[f'HR_{tag}_hi'] - ob[f'HR_{tag}'],
                                             arrayminus=ob[f'HR_{tag}'] - ob[f'HR_{tag}_lo'], color=col)))
    hr.add_vline(x=1, line_color=C['muted'], line_dash='dot')
    layout_fig(hr, 60 + 40 * len(ob), xaxis=dict(type='log', title='Observed hazard ratio for OS (log scale), 95% CI'))
    return html.Div([
        card([html.Div('In-silico flips across all random5 test patients (biology.py)', className='card-title'),
              dcc.Graph(figure=fig, config={'displayModeBar': False}),
              html.Div('Each row edits one token for every eligible NSCLC test landmark and re-scores it with its fold model. '
                       'Stage IV vs I is the positive control.', className='note')]),
        card([html.Div('What the data says: observed OS, one landmark per patient', className='card-title'),
              dcc.Graph(figure=hr, config={'displayModeBar': False})]),
    ])


def perf_tab():
    r = pd.concat([pd.read_csv(f) for f in ('results.csv', 'results_random5_hstu_B200.csv') if os.path.exists(f)])
    r = r[(r.split == 'random5') & (r.fold == 'POOLED')].drop_duplicates(['model', 'metric'], keep='last')
    figs = []
    for m in ('cindex', 'auc12', 'auc24'):
        d = r[r.metric == m].sort_values('value')
        fig = go.Figure(go.Bar(x=d.value, y=d.model, orientation='h', marker_color=[C['cf'] if x.startswith('HSTU') else C['base'] for x in d.model],
                               error_x=dict(type='data', symmetric=False, array=d.ci_hi - d.value, arrayminus=d.value - d.ci_lo, color=C['muted']),
                               text=[f'{v:.3f}' for v in d.value], textposition='inside', insidetextanchor='end'))
        layout_fig(fig, 60 + 36 * len(d), title=dict(text={'cindex': 'C-index', 'auc12': 'AUC 12 months', 'auc24': 'AUC 24 months'}[m], x=0),
                   xaxis=dict(range=[0.5, max(0.85, d.ci_hi.max() + 0.02)]))
        figs.append(card([dcc.Graph(figure=fig, config={'displayModeBar': False})], flex='1', minWidth='300px'))
    rows = []
    for f in sorted(x for x in os.listdir('.') if x.startswith('paired_') and x.endswith('.csv')):
        d = pd.read_csv(f); d = d[d.fold == 'POOLED']
        rows += [html.Tr([html.Td(f'{x.a} − {x.b}'), html.Td(f'{x.diff:+.3f}'), html.Td(f'[{x.ci_lo:+.3f}, {x.ci_hi:+.3f}]'),
                          html.Td(f'{x.n_landmarks:,}')]) for x in d.itertuples()]
    tbl = html.Table([html.Thead(html.Tr([html.Th(h) for h in ('Paired comparison (C-index)', 'Difference', '95% CI', 'Landmarks')])),
                      html.Tbody(rows)], className='tbl')
    return html.Div([html.Div(figs, className='row'),
                     card([html.Div('Paired patient-level bootstrap, pooled over the folds each comparison has so far', className='card-title'), tbl])])


app.layout = html.Div([
    html.Div([html.Div([html.Span('HSTU', className='logo'), html.Span(' trajectory simulator', className='title')]),
              html.Div('GENIE BPC · overall survival from each regimen start · random5 test patients only', className='subtitle')],
             className='header'),
    dcc.Tabs(id='tabs', value='sim', className='tabs', children=[
        dcc.Tab(label='Patient simulator', value='sim', className='tab', selected_className='tab-sel', children=sim_tab),
        dcc.Tab(label='Cohort checks', value='checks', className='tab', selected_className='tab-sel'),
        dcc.Tab(label='Model performance', value='perf', className='tab', selected_className='tab-sel'),
    ]),
    html.Div(id='tab-body'),
], className='page')


@app.callback(Output('tab-body', 'children'), Input('tabs', 'value'))
def tab_body(t):
    return checks_tab() if t == 'checks' else perf_tab() if t == 'perf' else None


@app.callback(Output('patient', 'options'), Output('patient', 'value'),
              Input('cohort', 'value'), Input('filter', 'value'), Input('random', 'n_clicks'), State('patient', 'value'))
def patient_options(cohort, flt, _, cur):
    d = lm[lm.cohort == cohort]
    d = d[d.stk_keap] if flt == 'stk' else d[d.immuno] if flt == 'io' else d[d.platinum] if flt == 'plat' else d
    n = lm[lm.patient_id.isin(set(d.patient_id))].groupby('patient_id').size()
    opts = [{'label': f'Patient {pat_no[p]} · {k} treatment point{"s" if k > 1 else ""}', 'value': p}
            for p, k in sorted(n.items(), key=lambda x: pat_no[x[0]])]
    if not opts: return [], None
    ids = [o['value'] for o in opts]
    if ctx.triggered_id == 'random' or cur not in ids:
        return opts, str(np.random.choice(ids))
    return opts, no_update


def patient_landmarks(pid):
    return lm[lm.patient_id == pid].sort_values(['landmark_day', 'landmark_id'])


@app.callback(Output('step', 'max'), Output('step', 'marks'), Output('step', 'value'), Input('patient', 'value'))
def steps(pid):
    if pid is None: return 0, {}, 0
    d = patient_landmarks(pid)
    marks = {i: {'label': f'L{x.line} · day {int(x.landmark_day)}'} for i, x in enumerate(d.itertuples())}
    return len(d) - 1, marks, 0


@app.callback(Output('landmark', 'data'), Input('patient', 'value'), Input('step', 'value'))
def pick_landmark(pid, i):
    if pid is None: return None
    d = patient_landmarks(pid)
    return int(d.landmark_id.iloc[min(i or 0, len(d) - 1)])


@app.callback(Output('patient-card', 'children'), Output('timeline', 'figure'),
              Output('stage', 'options'), Output('stage', 'value'), Output('regimen', 'options'), Output('regimen', 'value'),
              Output('remove', 'options'), Output('remove', 'value'), Output('tmb', 'value'), Output('tmb', 'disabled'),
              Output('add', 'value'), Output('mets', 'value'), Output('scan', 'value'), Output('md', 'value'),
              Input('landmark', 'data'), Input('reset', 'n_clicks'))
def show_patient(lid, _):
    if lid is None: return [no_update] * 14
    r, Fd, model, ix = landmark(lid)
    find = findings(ix)
    tmbs = [tok_str[j].split(':', 1)[1] for j in ix if etype_str[j] == 'TMB']
    panel = next((tok_str[j].split(':', 1)[1] for j in ix[::-1] if etype_str[j] == 'PANEL'), '–')
    died = r.os_event == 1
    chips = [html.Span(t, className='chip') for _, t in find] or [html.Span('no findings on panel', className='chip chip-muted')]
    info = [('Cohort', r.cohort), ('Histology', r.HIST), ('Stage', r.STAGE), ('Age', r.AGE), ('Sex', r.SEX),
            ('Line', r.line), ('Regimen', r.reg_name), ('Day since diagnosis', int(r.landmark_day)), ('Panel', panel),
            ('TMB', tmbs[-1] if tmbs else 'not reported'), ('Test fold', r.fold),
            ('Observed', f'{"died" if died else "alive (censored)"} at {r.os_time / 30.44:.1f} mo')]
    pc = card([html.Div([html.Div([html.Div(k, className='info-k'), html.Div(str(v), className='info-v')], className='info') for k, v in info],
                        className='infos'),
               html.Div([html.Span('Genomic findings ', className='ctl-label')] + chips, className='chips')], flex='1')
    # timeline
    fig = go.Figure(); lanes = [n for n, _ in LANES]
    ev = E.ev.iloc[ix]; ev = ev[ev.etype != 'STATIC']
    starts = ev[ev.etype == 'REG_START']; ends = ev[ev.etype == 'REG_END']
    for x in starts.itertuples():
        e = ends[ends.ref == x.ref].day
        end = e.iloc[0] if len(e) else r.landmark_day
        is_lm = x.Index == E.ev.index[ix[-1]]
        fig.add_trace(go.Scatter(x=[x.day, max(end, x.day + 3)], y=['Treatment'] * 2, mode='lines', showlegend=False,
                                 line=dict(color=C['cf'] if is_lm else LANE_COLOR['Treatment'], width=12), opacity=0.85,
                                 hovertemplate=f'{x.value}<br>day {int(x.day)}–{int(end)}<extra></extra>'))
    for name, es in LANES[1:]:
        d = ev[ev.etype.isin(es)]
        if len(d):
            fig.add_trace(go.Scatter(x=d.day, y=[name] * len(d), mode='markers', name=name, showlegend=False,
                                     marker=dict(size=9, color=LANE_COLOR[name], line=dict(width=1, color=C['bg'])),
                                     text=d.token, hovertemplate='%{text}<br>day %{x:.0f}<extra></extra>'))
    fig.add_vline(x=r.landmark_day, line_color=C['cf'], line_dash='dash', annotation_text='landmark', annotation_font_color=C['cf'])
    layout_fig(fig, 300, xaxis_title='Days since diagnosis', yaxis=dict(categoryorder='array', categoryarray=lanes[::-1]))
    regs = list(dict.fromkeys([r.reg_name] + reg_by_cohort[r.cohort]))
    regs = [x for x in regs if token_id(E, Fd, f'REG_START:{x}') is not None]
    return (pc, fig, stage_options(Fd), r.STAGE, regs, r.reg_name, [{'label': t, 'value': k} for k, t in find], [],
            tmbs[-1] if tmbs else None, not tmbs, [], [], None, None)


@app.callback(Output('kpis', 'children'), Output('survival', 'figure'), Output('trajectory', 'figure'),
              Input('landmark', 'data'), Input('stage', 'value'), Input('regimen', 'value'), Input('add', 'value'),
              Input('remove', 'value'), Input('tmb', 'value'), Input('mets', 'value'), Input('scan', 'value'), Input('md', 'value'))
def simulate(lid, stage, regimen, add_genes, remove_pos, tmb, mets, scan, md):
    if lid is None: return no_update, no_update, no_update
    r, Fd, model, ix = landmark(lid)
    s0 = arrays(E, Fd, ix); s1 = edited(r, Fd, ix, stage, regimen, add_genes, remove_pos, tmb, scan, md, mets)
    S0, S1 = survival(model, [s0, s1], DEVICE)
    # prognosis over time: every treatment point of this patient (same fold, so the same model), plus the what-if here
    d = patient_landmarks(r.patient_id)
    St = survival(model, [arrays(E, Fd, truncate(E, E.P[l])) for l in d.landmark_id.values], DEVICE)
    here = int(np.where(d.landmark_id.values == lid)[0][0])
    tr = go.Figure()
    tr.add_trace(go.Scatter(x=d.landmark_day, y=1 - St[:, 11], mode='lines+markers', name='12-month death risk (as observed)',
                            line=dict(color=C['base'], width=2), marker=dict(size=9),
                            text=[f'line {x.line} · {x.reg_name}' for x in d.itertuples()],
                            hovertemplate='day %{x:.0f}: %{y:.0%}<br>%{text}<extra></extra>'))
    tr.add_trace(go.Scatter(x=d.landmark_day, y=1 - St[:, 23], mode='lines+markers', name='24-month death risk',
                            line=dict(color=C['accent'], width=1.5, dash='dot'), marker=dict(size=6),
                            hovertemplate='day %{x:.0f}: %{y:.0%}<extra>24-month</extra>'))
    tr.add_trace(go.Scatter(x=[r.landmark_day], y=[1 - S1[11]], mode='markers', name='what-if (12-month)',
                            marker=dict(symbol='star', size=18, color=C['cf'], line=dict(width=1, color=C['bg']))))
    tr.add_vline(x=r.landmark_day, line_color=C['cf'], line_dash='dash')
    end = d.landmark_day.iloc[0] + d.os_time.iloc[0]
    tr.add_vline(x=end, line_color=C['muted'], line_dash='dot',
                 annotation_text='died' if d.os_event.max() == 1 else 'last seen alive', annotation_font_color=C['muted'])
    layout_fig(tr, 230, xaxis_title='Days since diagnosis', yaxis=dict(tickformat='.0%', range=[0, 1.02], title='Risk'))
    m = np.arange(0, 61)
    S0, S1 = np.r_[1, S0], np.r_[1, S1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=m, y=S0, name='as observed', line=dict(color=C['base'], width=2, dash='dash'),
                             hovertemplate='%{x} mo: %{y:.0%}<extra>as observed</extra>'))
    fig.add_trace(go.Scatter(x=m, y=S1, name='what-if', line=dict(color=C['cf'], width=3), fill='tonexty',
                             fillcolor='rgba(255,138,92,0.12)', hovertemplate='%{x} mo: %{y:.0%}<extra>what-if</extra>'))
    t = min(r.os_time / 30.44, 60)
    fig.add_trace(go.Scatter(x=[t], y=[float(np.interp(t, m, S0))], mode='markers', name='died' if r.os_event == 1 else 'last seen alive',
                             marker=dict(symbol='x' if r.os_event == 1 else 'circle-open', size=13, color=C['text'], line=dict(width=2))))
    for v in (12, 24): fig.add_vline(x=v, line_color=C['line'])
    layout_fig(fig, 340, xaxis_title='Months from regimen start', yaxis=dict(title='Predicted survival', tickformat='.0%', range=[0, 1.02]))
    pct = lambda v: f'{100 * v:.0f}%'
    k = [kpi('12-month death risk', 1 - S0[12], 1 - S1[12], pct), kpi('24-month death risk', 1 - S0[24], 1 - S1[24], pct),
         html.Div([html.Div('Predicted median OS', className='kpi-label'),
                   html.Div([html.Span(median_months(S0[1:]), className='kpi-base'), html.Span(' → ', className='kpi-arrow'),
                             html.Span(median_months(S1[1:]), className='kpi-value')])], className='kpi')]
    return k, fig, tr


if __name__ == '__main__':
    print(f'serving on 127.0.0.1:{a.port}, open <JupyterLab URL>{prefix.replace("/jupyterlab/default", "")}', flush=True)
    app.run(host='127.0.0.1', port=a.port, debug=False)
