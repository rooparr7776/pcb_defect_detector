"""PCB inspection V4.1 - classical image processing only (no ML / deep learning).
Run:  streamlit run nanonets_web_v4.py      (keep pcb_core.py in the same folder)
"""
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
import pcb_core_ as core

st.set_page_config(page_title='PCB Inspection V4.1', layout='wide')
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'NanoNetsPCBChecker' / 'Dataset'
FLAG = ('MISSING', 'ORIENTATION/PLACEMENT SUSPECT')


@st.cache_resource(show_spinner=False)
def build_library(folder, n_lib):
    paths = core.discover(folder)
    gold = [p for p in paths if core.is_golden(p)]
    if not gold:
        return None
    canon = next((p for p in gold if p.stem == '1'), gold[0])
    step = max(1, len(gold) // n_lib)
    chosen = list(dict.fromkeys(gold[::step][:n_lib] + [canon]))
    return core.Library(chosen, canon)


def pct(x):
    return 'N/A' if x is None else f'{x:.1%}'


def show_metrics(title, m, extra=''):
    c = st.columns(4)
    c[0].metric(title + ' precision', pct(m['Precision']))
    c[1].metric('Recall', pct(m['Recall']))
    c[2].metric('F1', pct(m['F1']))
    c[3].metric('TP / FP / FN', f"{m['TP']} / {m['FP']} / {m['FN']}")
    if extra:
        st.caption(extra)


st.title('PCB Component Inspection - Golden-Library Template Matching')
st.caption('Classical image processing only. XML gives the component boxes of the canonical board; '
           'the test XML is used only afterwards, for evaluation.')

with st.sidebar:
    folder = st.text_input('Dataset directory', str(DEFAULT))
    n_lib = st.slider('Golden boards in library', 5, 40, 20,
                      help='More boards = fewer false alarms but slower first start.')
    miss_thr = st.slider('MISSING if score below', 0.30, 0.90, 0.58, 0.01)
    defect_thr = st.slider('DEFECT / review if score below', 0.40, 0.95, 0.70, 0.01,
                           help='Sharp cliff above ~0.73 on this dataset: false alarms jump.')
    search = st.slider('Local search window (px @ half-scale)', 4, 40, 24)
    exclude_own = st.checkbox("Exclude the test board's own golden image", True,
                              help='Keep ON for an honest evaluation.')

paths = core.discover(folder)
if not paths:
    st.error('No image/XML pairs found. Point to NanoNetsPCBChecker/Dataset.')
    st.stop()
with st.spinner('Aligning golden boards (first run only)...'):
    lib = build_library(folder, n_lib)
if lib is None:
    st.error('No golden boards (60 components, none rotated) found.')
    st.stop()
st.info(f'Library: {len(lib.ch)} golden boards aligned to {lib.canonical_path.name}; '
        f'{len(lib.objects)} components.')

tab1, tab2 = st.tabs(['Single board', 'Batch evaluation (all N.N pairs)'])

with tab1:
    tests = [p for p in paths if not core.is_golden(p)]
    test = st.selectbox('Test PCB', tests, format_func=lambda p: p.name)
    if st.button('Run inspection', type='primary'):
        res = core.inspect(lib, test, miss_thr, defect_thr, search, exclude_own)
        frame = pd.DataFrame(res['rows']).drop(columns=['box'])
        overlay = core.draw(res['aligned'], res['rows'])
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(res['test'], cv2.COLOR_BGR2RGB), caption='Test board', use_container_width=True)
        c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Aligned (ECC {res['ecc']:.3f}). Red = missing, orange = orientation/placement suspect, green = OK",
                 use_container_width=True)
        if not res['ecc_ok']:
            st.warning('ECC alignment failed - results are unreliable.')
        st.subheader('Flagged components')
        st.dataframe(frame[frame.Status.isin(FLAG)].sort_values('Score'), hide_index=True, use_container_width=True)
        with st.expander('All components'):
            st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button('Download CSV', frame.to_csv(index=False).encode(),
                           file_name=f'pcb_{test.stem}.csv', mime='text/csv')
        st.subheader('Evaluation against XML')
        missing, rotated = core.ground_truth(lib, test)
        defects = missing | rotated
        pm = set(frame.loc[frame.Status == 'MISSING', 'Component'])
        pd_ = set(frame.loc[frame.Status.isin(FLAG), 'Component'])
        show_metrics('Defect (missing or rotated)', core.metrics(defects, pd_),
                     f"Actual: {', '.join(sorted(defects)) or 'none'} | flagged: {', '.join(sorted(pd_)) or 'none'} | "
                     f"component accuracy {core.accuracy(lib, res['rows'], defects):.1%}")
        show_metrics('Missing-only', core.metrics(missing, pm),
                     f"Actual missing: {', '.join(sorted(missing)) or 'none'} | predicted MISSING: {', '.join(sorted(pm)) or 'none'}")

with tab2:
    st.write('Runs every test board named N.N.jpg and reports overall precision / recall / accuracy.')
    if st.button('Run batch evaluation'):
        pairs = [p for p in paths if '.' in p.stem and p.stem.split('.')[0] == p.stem.split('.')[1]]
        agg = {k: dict(TP=0, FP=0, FN=0) for k in ('defect', 'missing')}
        correct = total = 0
        out = []
        bar = st.progress(0.0)
        for i, p in enumerate(pairs):
            res = core.inspect(lib, p, miss_thr, defect_thr, search, exclude_own)
            miss, rot = core.ground_truth(lib, p)
            dfc = miss | rot
            pm = {r['Component'] for r in res['rows'] if r['Status'] == 'MISSING'}
            pf = {r['Component'] for r in res['rows'] if r['Status'] in FLAG}
            for key, a, b in (('defect', dfc, pf), ('missing', miss, pm)):
                m = core.metrics(a, b)
                for k in ('TP', 'FP', 'FN'):
                    agg[key][k] += m[k]
            n = len([r for r in res['rows'] if not r['Status'].startswith('SKIPPED')])
            correct += n - len(pf ^ dfc); total += n
            out.append({'Board': p.stem, 'Actual defects': ', '.join(sorted(dfc)),
                        'Flagged': ', '.join(sorted(pf)), 'Predicted MISSING': ', '.join(sorted(pm))})
            bar.progress((i + 1) / len(pairs))
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
        for key, title in (('defect', 'Defect detection (missing or rotated)'), ('missing', 'Missing-only (strict)')):
            t = agg[key]
            P = t['TP'] / (t['TP'] + t['FP']) if t['TP'] + t['FP'] else None
            R = t['TP'] / (t['TP'] + t['FN']) if t['TP'] + t['FN'] else None
            F = 2 * P * R / (P + R) if P and R else None
            st.markdown(f"**{title}** - TP={t['TP']}, FP={t['FP']}, FN={t['FN']} | "
                        f"precision {pct(P)}, recall {pct(R)}, F1 {pct(F)}")
        st.markdown(f'**Component-level accuracy:** {correct / max(1, total):.2%} ({correct}/{total} decisions)')
