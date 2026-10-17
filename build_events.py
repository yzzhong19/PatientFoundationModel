"""Step 1/2: build events.parquet, landmarks.parquet, genomics_status.parquet from GENIE BPC raw files.

Clock: every timeline table has a birth-based integer day column (drugs_startdt_int_*, image_scan_int,
md_onc_visit_int, dob_cpt_report_days, tm_spec_collect_int, path_proc_int). We re-zero all of them to the
patient's first index-cancer diagnosis (dob_ca_dx_days), so day 0 == diagnosis for every table.
Outcome columns (death, last contact, os_*/pfs_* columns) are used ONLY to build landmarks.parquet labels.
"""
import glob, os, re, sys, collections
import numpy as np, pandas as pd

RAW = os.path.expanduser('~/hstu/raw')
OUT = os.path.expanduser('~/hstu')
COHORTS = {'NSCLC': 'NSCLC', 'CRC': 'CRC', 'BrCa': 'Breast', 'PANC': 'Pancreatic', 'Prostate': 'Prostate', 'BLADDER': 'Bladder'}
NONSILENT = {'Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 'Frame_Shift_Ins', 'In_Frame_Del',
             'In_Frame_Ins', 'Splice_Site', 'Translation_Start_Site', 'Nonstop_Mutation'}
# fixed within-day order; static/panel first, REG_START last so the landmark token is the last of its day
RANK = {'STATIC': 0, 'PANEL': 1, 'SAMPLE': 2, 'TMB': 3, 'MUT': 4, 'FUSION': 5, 'AMP': 6, 'DEL': 7, 'PATH': 8,
        'TM': 9, 'MET': 10, 'MD': 11, 'SCAN': 12, 'REG_END': 13, 'REG_START': 14}
CLASSES = ['chemo', 'immuno', 'targeted', 'hormonal', 'antiangio']
IMMUNO = 'pembrolizumab nivolumab atezolizumab durvalumab ipilimumab avelumab tremelimumab bcg aldesleukin interferon peginterferon sipuleucel talimogene'
TARGETED = ('erlotinib gefitinib afatinib osimertinib crizotinib alectinib brigatinib ceritinib lorlatinib entrectinib larotrectinib '
            'dabrafenib trametinib vemurafenib cobimetinib binimetinib encorafenib lapatinib neratinib tucatinib trastuzumab pertuzumab '
            'cetuximab panitumumab necitumumab olaparib rucaparib talazoparib palbociclib ribociclib abemaciclib everolimus temsirolimus '
            'alpelisib taselisib erdafitinib imatinib dasatinib nilotinib bosutinib ponatinib ibrutinib acalabrutinib venetoclax '
            'rociletinib icotinib ensartinib sacituzumab enfortumab rituximab obinutuzumab bortezomib carfilzomib ixazomib daratumumab '
            'elotuzumab lenalidomide pomalidomide thalidomide ruxolitinib navitoclax brentuximab ').split()
HORMONAL = ('abiraterone enzalutamide apalutamide darolutamide bicalutamide flutamide nilutamide leuprolide goserlin goserelin degarelix '
            'triptorelin histrelin adt tamoxifen toremifene fulvestrant anastrozole letrozole exemestane megestrol dutasteride '
            'ketoconazole estramustine other hormone').split()
ANTIANGIO = 'bevacizumab ramucirumab ziv sunitinib sorafenib pazopanib cabozantinib lenvatinib regorafenib apatinib'.split()
IMMUNO = IMMUNO.split()
NONCHEMO_MISC = {'investigational', 'other'}


def drug_key(s):
    return re.split(r'[\(]', s)[0].strip().lower()


def drug_class(name):
    k = drug_key(name)
    first = k.split()[0] if k else ''
    m = 0
    if any(first.startswith(x) for x in IMMUNO) or k.startswith('bcg'): m |= 2
    elif any(first.startswith(x) for x in ANTIANGIO): m |= 16
    elif any(first.startswith(x) or k.startswith(x) for x in TARGETED): m |= 4
    elif any(first.startswith(x) or k.startswith(x) for x in HORMONAL): m |= 8
    elif first in NONCHEMO_MISC or k.startswith('other nos'): m |= 0
    else: m |= 1  # remaining antineoplastics are cytotoxic chemotherapy
    return m


def read(path, **kw):
    return pd.read_csv(path, low_memory=False, **kw)


def panel_genes(cb):
    out = {}
    for f in glob.glob(f'{cb}/data_gene_panel_*.txt'):
        aid = os.path.basename(f)[len('data_gene_panel_'):-4]
        for line in open(f):
            if line.startswith('gene_list:'):
                out[aid] = set(line.rstrip('\n').split('\t')[1:])
    return out


def build_cohort(coh, folder, top_genes=None):
    base = glob.glob(f'{RAW}/{coh}/*/')[0]
    cd = glob.glob(f'{base}*clinical_data')[0]
    cb = f'{base}cBioPortal_files'
    cohort = folder
    pat = read(f'{cd}/patient_level_dataset.csv')
    can = read(f'{cd}/cancer_level_dataset_index.csv')
    can = can[can.redcap_ca_index == 'Yes'].sort_values(['record_id', 'dob_ca_dx_days'])
    first = can.drop_duplicates('record_id').set_index('record_id')
    anchor = first.dob_ca_dx_days
    pid = lambda s: s.map(lambda x: x)  # noqa
    ev = []  # list of DataFrames: patient_id, day, etype, value, dclass, ref

    def add(df, etype, value, dclass=0, ref=''):
        d = pd.DataFrame({'patient_id': df['patient_id'].values, 'day': df['day'].values.astype(float),
                          'etype': etype, 'value': value if not isinstance(value, pd.Series) else value.values,
                          'dclass': dclass if not isinstance(dclass, pd.Series) else dclass.values,
                          'ref': ref if not isinstance(ref, pd.Series) else ref.values})
        ev.append(d)

    # ---- static
    st = first.join(pat.set_index('record_id')[['institution', 'naaccr_sex_code']], rsuffix='_p')
    st = st.reset_index().rename(columns={'record_id': 'patient_id'})
    st['day'] = 0
    hist = st.ca_hist_adeno_squamous.fillna(st.ca_type).fillna('NA').astype(str)
    age = pd.cut(st.age_dx.astype(float), [0, 40, 50, 60, 70, 80, 200], labels=['<40', '40s', '50s', '60s', '70s', '80+']).astype(str)
    stage = st.stage_dx.fillna('NA').astype(str)
    for et, v in [('COHORT', cohort), ('HIST', hist), ('STAGE', stage), ('AGE', age), ('SEX', st.naaccr_sex_code.fillna('NA').astype(str))]:
        add(st, 'STATIC', ('%s=' % et) + (v if isinstance(v, pd.Series) else pd.Series(v, index=st.index)))
    inst = dict(zip(st.patient_id, st.institution))

    def rel(df, col, name='day'):
        df = df.rename(columns={'record_id': 'patient_id'})
        df = df[df.patient_id.isin(anchor.index)].copy()
        df['day'] = df[col] - df.patient_id.map(anchor)
        return df.dropna(subset=['day'])

    # ---- regimens (index cancers only; drug start cols are birth-based)
    reg = read(f'{cd}/regimen_cancer_level_dataset.csv')
    reg = reg[reg.redcap_ca_index == 'Yes']
    sc = [f'drugs_startdt_int_{k}' for k in range(1, 6)]
    reg['start_abs'] = reg[sc].min(axis=1)
    reg['end_abs'] = reg.start_abs + (reg.dx_reg_end_all_int - reg.dx_reg_start_int)
    dcols = [f'drugs_drug_{k}' for k in range(1, 6)]
    reg['names'] = reg[dcols].apply(lambda r: '+'.join(sorted({drug_key(x).split()[0] if drug_key(x) else '' for x in r.dropna()})), axis=1)
    reg['dclass'] = reg[dcols].apply(lambda r: int(np.bitwise_or.reduce([drug_class(x) for x in r.dropna()] or [0])), axis=1)
    reg['ref'] = reg.ca_seq.astype(str) + '_' + reg.regimen_number.astype(str)
    rs = rel(reg, 'start_abs')
    add(rs, 'REG_START', rs.names, rs.dclass, rs.ref)
    re_ = rel(reg.dropna(subset=['end_abs']), 'end_abs')
    add(re_, 'REG_END', re_.names, re_.dclass, re_.ref)

    # ---- imaging / medonc
    im = rel(read(f'{cd}/imaging_level_dataset.csv'), 'image_scan_int')
    smap = {'Progressing/Worsening/Enlarging': 'progressing', 'Stable/No change': 'stable', 'Improving/Responding': 'improving', 'Mixed': 'mixed'}
    im['v'] = im.image_overall.map(smap)
    im = im.dropna(subset=['v']).drop_duplicates(['patient_id', 'day', 'v'])
    add(im, 'SCAN', im.v)
    md = rel(read(f'{cd}/med_onc_note_level_dataset.csv'), 'md_onc_visit_int')
    md['v'] = md.md_ca_status.map(lambda s: {'Progressing/Worsening/Enlarging': 'progressing', 'Stable/No change': 'stable', 'Improving/Responding': 'improving', 'Mixed': 'mixed', 'Not stated/Indeterminate': 'indet'}.get(s))
    md = md.dropna(subset=['v']).drop_duplicates(['patient_id', 'day', 'v'])
    add(md, 'MD', md.v)

    # ---- metastasis sites (first documented date per site, dx-relative columns of the anchor cancer)
    sites = [c[len('dx_to_dmets_'):-5] for c in can.columns if re.fullmatch(r'dx_to_dmets_[a-z_]+_days', c) and c != 'dx_to_dmets_days']
    f2 = first.reset_index().rename(columns={'record_id': 'patient_id'})
    for s in sites:
        d = f2[['patient_id', f'dx_to_dmets_{s}_days']].dropna().rename(columns={f'dx_to_dmets_{s}_days': 'day'})
        add(d, 'MET', s)

    # ---- tumor markers (bucket by ratio to upper limit of normal; PSA by absolute)
    for f in glob.glob(f'{cd}/tm_level_dataset.csv'):
        tm = rel(read(f), 'tm_spec_collect_int')
        tm = tm[tm.tm_type.isin(['CEA', 'CA19-9', 'PSA', 'CA15-3', 'CA2729'])].dropna(subset=['tm_num_result'])
        ratio = tm.tm_num_result / tm.tm_normal_range_upper.replace(0, np.nan)
        ratio = ratio.where(ratio.notna(), tm.tm_num_result / tm.tm_type.map({'CEA': 5., 'CA19-9': 37., 'PSA': 4., 'CA15-3': 30., 'CA2729': 38.}))
        b = pd.cut(ratio, [-np.inf, 1, 2, 5, 20, np.inf], labels=['norm', '1-2x', '2-5x', '5-20x', '>20x']).astype(str)
        tm = tm.assign(v=tm.tm_type + '=' + b).drop_duplicates(['patient_id', 'day', 'v'])
        add(tm, 'TM', tm.v)

    # ---- pathology: procedure type
    pa = rel(read(f'{cd}/pathology_report_level_dataset.csv'), 'path_proc_int')
    pa = pa.dropna(subset=['path_proc_type']).drop_duplicates(['patient_id', 'day', 'path_proc_type'])
    add(pa, 'PATH', pa.path_proc_type.astype(str).str.slice(0, 30))

    # ---- genomics
    cpt = rel(read(f'{cd}/cancer_panel_test_level_dataset.csv'), 'dob_cpt_report_days')
    cpt = cpt[cpt.cpt_report_post_death.fillna(0) == 0]
    pg = panel_genes(cb)
    cpt = cpt[cpt.cpt_seq_assay_id.isin(pg.keys())].copy()
    s2p = dict(zip(cpt.cpt_genie_sample_id, cpt.patient_id))
    s2day = dict(zip(cpt.cpt_genie_sample_id, cpt.day))
    s2assay = dict(zip(cpt.cpt_genie_sample_id, cpt.cpt_seq_assay_id))
    add(cpt, 'PANEL', cpt.cpt_seq_assay_id)
    cs = cpt.dropna(subset=['sample_type'])
    add(cs, 'SAMPLE', cs.sample_type.astype(str).str.slice(0, 30))
    mut = pd.read_csv(f'{cb}/data_mutations_extended.txt', sep='\t', usecols=['Hugo_Symbol', 'Variant_Classification', 'Tumor_Sample_Barcode'], low_memory=False)
    mut = mut[mut.Tumor_Sample_Barcode.isin(s2p)]
    mut = mut[mut.Variant_Classification.isin(NONSILENT)]
    # TMB = non-silent coding mutations / number of genes on the panel (panel-size normalised)
    cnt = mut.groupby('Tumor_Sample_Barcode').size()
    tmbdf = pd.DataFrame({'sample': list(s2p)})
    tmbdf['n'] = tmbdf['sample'].map(cnt).fillna(0)
    tmbdf['ng'] = tmbdf['sample'].map(lambda s: len(pg[s2assay[s]]))
    tmbdf['tmb'] = tmbdf.n / tmbdf.ng * 100
    tmbdf['patient_id'] = tmbdf['sample'].map(s2p); tmbdf['day'] = tmbdf['sample'].map(s2day)
    mutd = mut.drop_duplicates(['Hugo_Symbol', 'Tumor_Sample_Barcode']).copy()
    mutd['patient_id'] = mutd.Tumor_Sample_Barcode.map(s2p); mutd['day'] = mutd.Tumor_Sample_Barcode.map(s2day)
    # fusions
    fus = []
    if os.path.exists(f'{cb}/data_fusions.txt'):
        f = pd.read_csv(f'{cb}/data_fusions.txt', sep='\t', usecols=['Hugo_Symbol', 'Tumor_Sample_Barcode', 'Fusion'], low_memory=False)
        f = f[~f.Fusion.astype(str).str.contains('intragenic')]
        fus.append(f.rename(columns={'Tumor_Sample_Barcode': 'sample', 'Hugo_Symbol': 'gene'})[['sample', 'gene']])
    if os.path.exists(f'{cb}/data_sv.txt'):
        f = pd.read_csv(f'{cb}/data_sv.txt', sep='\t', usecols=['Sample_Id', 'Site1_Hugo_Symbol', 'Site2_Hugo_Symbol', 'SV_Status'], low_memory=False)
        f = f[(f.SV_Status == 'SOMATIC') & f.Site2_Hugo_Symbol.notna() & (f.Site1_Hugo_Symbol != f.Site2_Hugo_Symbol)]
        fus.append(pd.concat([f[['Sample_Id', 'Site1_Hugo_Symbol']].set_axis(['sample', 'gene'], axis=1), f[['Sample_Id', 'Site2_Hugo_Symbol']].set_axis(['sample', 'gene'], axis=1)]))
    fus = pd.concat(fus).dropna().drop_duplicates() if fus else pd.DataFrame(columns=['sample', 'gene'])
    fus = fus[fus['sample'].isin(s2p)].copy()
    fus['patient_id'] = fus['sample'].map(s2p); fus['day'] = fus['sample'].map(s2day)
    # CNA (+-2)
    cna = pd.read_csv(f'{cb}/data_CNA.txt', sep='\t', index_col=0, low_memory=False)
    cna = cna.loc[:, cna.columns.isin(s2p)]
    cl = cna.stack(); cl = cl[cl.isin([-2, 2])].reset_index(); cl.columns = ['gene', 'sample', 'val']
    cl['patient_id'] = cl['sample'].map(s2p); cl['day'] = cl['sample'].map(s2day)
    G = dict(mut=mutd.rename(columns={'Hugo_Symbol': 'gene'})[['patient_id', 'day', 'gene', 'Tumor_Sample_Barcode']].rename(columns={'Tumor_Sample_Barcode': 'sample'}),
             fus=fus, cna=cl, tmb=tmbdf, cpt=cpt[['patient_id', 'day', 'cpt_genie_sample_id', 'cpt_seq_assay_id']], pg=pg)
    # labels
    lab = reg.copy()
    lab['patient_id'] = lab.record_id
    lab = lab[lab.patient_id.isin(anchor.index)]
    lab['landmark_day'] = lab.start_abs - lab.patient_id.map(anchor)
    lab['cohort'] = cohort
    lab['institution'] = lab.patient_id.map(inst)
    lab = lab.merge(pat[['record_id', 'hybrid_death_int', 'dob_lastalive_int']].rename(columns={'record_id': 'patient_id'}), on='patient_id', how='left')
    lab['anchor_abs'] = lab.patient_id.map(anchor)
    lab['line'] = lab.groupby('patient_id').cumcount() + 1  # ordinal among all index regimens in time order (sorted below)
    lab = lab.sort_values(['patient_id', 'landmark_day'])
    lab['line'] = lab.groupby('patient_id').cumcount() + 1
    return pd.concat(ev, ignore_index=True).assign(cohort=cohort), G, lab, st


def main():
    cohorts = {}
    for coh, name in COHORTS.items():
        print('building', coh, flush=True)
        cohorts[name] = build_cohort(coh, name)
    # global top-300 genes by #patients with non-silent mutation (label-free frequency)
    allmut = pd.concat([G['mut'].assign(c=n) for n, (_, G, _, _) in cohorts.items()])
    top = allmut.drop_duplicates(['patient_id', 'gene']).gene.value_counts().head(300).index
    topset = set(top)
    pd.Series(sorted(topset)).to_csv(f'{OUT}/top_genes.csv', index=False, header=['gene'])
    evs, labs, status = [], [], []
    for name, (ev, G, lab, st) in cohorts.items():
        gev = []
        def addg(df, et, vals):
            if len(df): gev.append(pd.DataFrame({'patient_id': df.patient_id.values, 'day': df.day.values.astype(float), 'etype': et, 'value': vals, 'dclass': 0, 'ref': '', 'cohort': name}))
        m = G['mut'][G['mut'].gene.isin(topset)]; addg(m, 'MUT', m.gene.values)
        f = G['fus'][G['fus'].gene.isin(topset)]; addg(f, 'FUSION', f.gene.values)
        c = G['cna'][G['cna'].gene.isin(topset)]
        a = c[c.val == 2]; addg(a, 'AMP', a.gene.values)
        d = c[c.val == -2]; addg(d, 'DEL', d.gene.values)
        gev_tmb = G['tmb']
        evs.append(pd.concat([ev] + gev, ignore_index=True))
        # per-sample gene status: 1 mut/alt, 0 wild-type (tested), -1 untested (gene not on the sample's panel)
        samples = G['cpt'].rename(columns={'cpt_genie_sample_id': 'sample', 'cpt_seq_assay_id': 'assay'})
        mutset = set(zip(G['mut']['sample'], G['mut'].gene))
        rows = []
        for s, p, day, assay in samples[['sample', 'patient_id', 'day', 'assay']].itertuples(index=False):
            pgset = G['pg'][assay]
            rows.append((s, p, day, assay, name, {g: (1 if (s, g) in mutset else 0) if g in pgset else -1 for g in top}))
        status.append(pd.DataFrame(rows, columns=['sample', 'patient_id', 'day', 'assay', 'cohort', 'genes']))
        evs[-1] = pd.concat([evs[-1], tmb_events(G['tmb'], name)], ignore_index=True)
        labs.append(lab)
    ev = pd.concat(evs, ignore_index=True)
    # a few record_ids occur in two cohorts (second primary): drop them so no individual can straddle a split
    multi = ev.groupby('patient_id').cohort.nunique()
    multi = set(multi[multi > 1].index)
    print('dropping patients present in >1 cohort:', len(multi))
    ev = ev[~ev.patient_id.isin(multi)]
    # TMB buckets from label-free global quantiles of the per-sample TMB metric
    ev['rank'] = ev.etype.map(RANK)
    ev = ev[ev['rank'].notna()]
    ev['rank'] = ev['rank'].astype(int)
    # static tokens are placed at the earliest day of the patient
    mn = ev[ev.etype != 'STATIC'].groupby('patient_id').day.min()
    sm = ev.etype == 'STATIC'
    ev.loc[sm, 'day'] = ev.loc[sm, 'patient_id'].map(mn).fillna(0).clip(upper=0)
    ev['token'] = ev.etype + ':' + ev.value.astype(str)
    ev = ev.sort_values(['patient_id', 'day', 'rank', 'token'], kind='stable').reset_index(drop=True)
    ev = ev.drop_duplicates(['patient_id', 'day', 'token', 'ref'])
    ev['dclass'] = ev.dclass.astype('int16')
    ev[['patient_id', 'cohort', 'day', 'rank', 'etype', 'value', 'token', 'dclass', 'ref']].to_parquet(f'{OUT}/events.parquet', index=False)
    status = [d[~d.patient_id.isin(multi)] for d in status]
    pd.concat(status).assign(genes=lambda d: d.genes.map(lambda x: np.array([x[g] for g in sorted(topset)], dtype=np.int8))).to_pickle(f'{OUT}/genomics_status.pkl')
    lab = pd.concat(labs, ignore_index=True)
    lab = lab[~lab.patient_id.isin(multi)]
    lab.to_pickle(f'{OUT}/labels_raw.pkl')
    print(ev.groupby('etype').size())


_tmb_edges = None


def tmb_events(t, name):
    # fixed edges (mutations per 100 panel genes): chosen label-free, same for all cohorts
    b = pd.cut(t.tmb, [-1, 0.5, 1.5, 3, 6, 1e9], labels=['0-0.5', '0.5-1.5', '1.5-3', '3-6', '>6']).astype(str)
    return pd.DataFrame({'patient_id': t.patient_id.values, 'day': t.day.values.astype(float), 'etype': 'TMB', 'value': b.values, 'dclass': 0, 'ref': '', 'cohort': name})


if __name__ == '__main__':
    main()
