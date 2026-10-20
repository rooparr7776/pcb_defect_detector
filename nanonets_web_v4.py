"""PCB inspection V4 - classical image processing only (no ML / deep learning).
Run:  streamlit run nanonets_web_v4.py      (keep pcb_core.py in the same folder)
"""
from pathlib import Path
import cv2
import pandas as pd
import streamlit as st
import pcb_core as core

st.set_page_config(page_title='PCB Inspection V4', layout='wide')
ROOT = Path(__file__).resolve().parent
DEFAULT = ROOT / 'NanoNetsPCBChecker' / 'Dataset'


@st.cache_resource(show_spinner=False)
def build_library(folder, n_lib):
    paths = core.discover(folder)
    gold = [p for p in paths if core.is_golden(p)]
    if not gold:
        return None
    canon = next((p for p in gold if p.stem == '1'), gold[0])
    step = max(1, len(gold) // n_lib)
    chosen = list(dict.fromkeys(gold[::step][:n_lib] + [canon]))
    lib = core.Library(chosen, canon)
    lib.baseline()                       # pre-compute per-component normal range
    return lib


def pct(x):
    return 'N/A' if x is None else f'{x:.1%}'


st.title('PCB Component Inspection - Golden-Library Template Matching')
st.caption('Classical image processing only. XML gives the component boxes of the canonical '
           'board; the test XML is used only afterwards to evaluate.')

with st.sidebar:
    folder = st.text_input('Dataset directory', str(DEFAULT))
    n_lib = st.slider('Golden boards in library', 5, 40, 20,
                      help='More boards = fewer false alarms but slower start-up.')
    miss_thr = st.slider('Missing threshold (x component normal)', 0.30, 0.90, 0.55, 0.01)
    orient_thr = st.slider('Orientation/placement review threshold', 0.50, 0.95, 0.75, 0.01)
    search = st.slider('Local search window (px @ half-scale)', 4, 40, 24)
    exclude_own = st.checkbox('Exclude the test board\'s own golden image', True,
                              help='Keep ON for an honest evaluation (no peeking at its own reference).')

paths = core.discover(folder)
if not paths:
    st.error('No image/XML pairs found. Point to NanoNetsPCBChecker/Dataset.')
    st.stop()
with st.spinner('Aligning golden boards (first run only)...'):
    lib = build_library(folder, n_lib)
if lib is None:
    st.error('No golden boards (60 components, none rotated) found.')
    st.stop()
st.info(f'Library: {len(lib.gray)} golden boards aligned to {lib.canonical_path.name}; '
        f'{len(lib.objects)} components.')

tab1, tab2 = st.tabs(['Single board', 'Batch evaluation (all N.N pairs)'])

with tab1:
    tests = [p for p in paths if not core.is_golden(p)]
    test = st.selectbox('Test PCB', tests, format_func=lambda p: p.name)
    if st.button('Run inspection', type='primary'):
        res = core.inspect(lib, test, miss_thr, orient_thr, search, exclude_own)
        frame = pd.DataFrame(res['rows']).drop(columns=['box'])
        overlay = core.draw(res['aligned'], res['rows'])
        c1, c2 = st.columns(2)
        c1.image(cv2.cvtColor(res['test'], cv2.COLOR_BGR2RGB), caption='Test board', use_container_width=True)
        c2.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB),
                 caption=f"Aligned (ECC {res['ecc']:.3f}). Red = missing, orange = orientation/placement review, green = OK",
                 use_container_width=True)
        if not res['ecc_ok']:
            st.warning('ECC alignment failed - results are unreliable.')
        flagged = frame[frame['Status'] != 'OK'].sort_values('Score')
        st.subheader('Flagged components')
        st.dataframe(flagged, hide_index=True, use_container_width=True)
        with st.expander('All components'):
            st.dataframe(frame, hide_index=True, use_container_width=True)
        st.download_button('Download CSV', frame.to_csv(index=False).encode(),
                           file_name=f'pcb_{test.stem}.csv', mime='text/csv')
        st.subheader('Evaluation against XML')
        missing, rotated = core.ground_truth(lib, test)
        pm = set(frame.loc[frame.Status == 'MISSING', 'Component'])
        po = set(frame.loc[frame.Status == 'ORIENTATION/PLACEMENT SUSPECT', 'Component'])
        for title, act, pred in [('Missing', missing, pm), ('Rotated (review band, also catches some mis-placed parts)', rotated, po)]:
            m = core.metrics(act, pred)
            st.markdown(f'**{title}**: precision {pct(m["Precision"])}, recall {pct(m["Recall"])} '
                        f'(TP={m["TP"]}, FP={m["FP"]}, FN={m["FN"]})')
            st.write('XML truth: ' + (', '.join(sorted(act)) or 'none') +
                     ' | predicted: ' + (', '.join(sorted(pred)) or 'none'))

with tab2:
    st.write('Runs every test board named N.N.jpg and reports overall precision/recall.')
    if st.button('Run batch evaluation'):
        pairs = [p for p in paths if core.board_id(p) == p.stem.split('.')[-1] and '.' in p.stem
                 and p.stem.split('.')[0] == p.stem.split('.')[1]]
        tot = {'m': dict(TP=0, FP=0, FN=0), 'r': dict(TP=0, FP=0, FN=0)}
        out = []
        bar = st.progress(0.0)
        for i, p in enumerate(pairs):
            res = core.inspect(lib, p, miss_thr, orient_thr, search, exclude_own)
            miss, rot = core.ground_truth(lib, p)
            pm = {r['Component'] for r in res['rows'] if r['Status'] == 'MISSING'}
            po = {r['Component'] for r in res['rows'] if r['Status'] == 'ORIENTATION/PLACEMENT SUSPECT'}
            a, b = core.metrics(miss, pm), core.metrics(rot, po)
            for k in ('TP', 'FP', 'FN'):
                tot['m'][k] += a[k]; tot['r'][k] += b[k]
            out.append({'Board': p.stem, 'Actual missing': ', '.join(sorted(miss)),
                        'Predicted missing': ', '.join(sorted(pm)),
                        'Actual rotated': ', '.join(sorted(rot)), 'Review band': ', '.join(sorted(po))})
            bar.progress((i + 1) / len(pairs))
        st.dataframe(pd.DataFrame(out), hide_index=True, use_container_width=True)
        for key, title in (('m', 'Missing components'), ('r', 'Rotated components (review band)')):
            t = tot[key]
            p_ = t['TP'] / (t['TP'] + t['FP']) if t['TP'] + t['FP'] else None
            r_ = t['TP'] / (t['TP'] + t['FN']) if t['TP'] + t['FN'] else None
            st.markdown(f'**{title}** - TP={t["TP"]}, FP={t["FP"]}, FN={t["FN"]}, '
                        f'precision {pct(p_)}, recall {pct(r_)}')
