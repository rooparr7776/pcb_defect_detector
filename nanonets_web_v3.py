"""NanoNets PCB inspection: classical CV, no ML, XML used for evaluation only."""
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from skimage.metrics import structural_similarity as ssim

st.set_page_config(page_title='PCB Inspection V3',layout='wide')
ROOT=Path(__file__).resolve().parent
DEFAULT=ROOT/'NanoNetsPCBChecker'/'Dataset'
EXT={'.jpg','.jpeg','.png','.bmp'}

@st.cache_data(show_spinner=False)
def discover(folder):
    p=Path(folder)
    if not p.exists(): return []
    return sorted([str(f) for f in p.rglob('*') if f.suffix.lower() in EXT and f.with_suffix('.xml').exists() and 'errors' not in [v.lower() for v in f.parts]],key=lambda x:(Path(x).stem.split('.')[0],Path(x).name))

@st.cache_data(show_spinner=False)
def load(path):
    a=cv2.imread(path)
    return a

@st.cache_data(show_spinner=False)
def annotations(path):
    root=ET.parse(str(Path(path).with_suffix('.xml'))).getroot(); result={}
    for obj in root.findall('object'):
        name=(obj.findtext('name') or '').strip(); b=obj.find('bndbox')
        if not name or b is None: continue
        try:
            box=tuple(int(float(b.findtext(t))) for t in ('xmin','ymin','xmax','ymax'))
            rotated=int(float(obj.findtext('rotated','0') or 0))
        except (ValueError,TypeError): continue
        if box[2]>box[0] and box[3]>box[1]:result[name]={'box':box,'rotated':rotated}
    return result

def pair_key(path):
    return Path(path).stem.split('.')[0]

def clipped(a,box,pad=0):
    h,w=a.shape[:2];x1,y1,x2,y2=box
    return a[max(0,y1-pad):min(h,y2+pad),max(0,x1-pad):min(w,x2+pad)]

def register(ref,test):
    h,w=ref.shape[:2]
    if test.shape[:2]!=(h,w):test=cv2.resize(test,(w,h))
    # Work on downscaled grayscale to reduce memory and processing time.
    scale=min(1.0,1100/max(h,w));size=(max(100,int(w*scale)),max(100,int(h*scale)))
    r=cv2.resize(cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY),size).astype('float32')/255
    t=cv2.resize(cv2.cvtColor(test,cv2.COLOR_BGR2GRAY),size).astype('float32')/255
    warp=np.eye(2,3,dtype=np.float32)
    try:
        corr,warp=cv2.findTransformECC(r,t,warp,cv2.MOTION_AFFINE,(cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT,120,1e-5),None,5)
        warp[:,2]/=scale
        aligned=cv2.warpAffine(test,warp,(w,h),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=cv2.BORDER_REPLICATE)
        return aligned,float(corr),True
    except cv2.error:
        return test,0.,False

def normalized_gray(a):
    g=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY) if a.ndim==3 else a
    return cv2.GaussianBlur(g,(3,3),0)

def local_match(ref,test,box,search=8):
    # Local correction around each reference bounding box after global ECC.
    x1,y1,x2,y2=box
    h,w=ref.shape[:2]
    template=clipped(ref,box)
    if template.size==0:return None,None,0.
    th,tw=template.shape[:2]
    sx=max(0,x1-search);sy=max(0,y1-search)
    ex=min(w,x2+search);ey=min(h,y2+search)
    area=test[sy:ey,sx:ex]
    if area.shape[0]<th or area.shape[1]<tw:return template,clipped(test,box),0.
    # Grayscale normalized correlation; do not use this score as a defect score.
    tg=normalized_gray(template);ag=normalized_gray(area)
    if np.std(tg)<3: return template,clipped(test,box),0.
    corr=cv2.matchTemplate(ag,tg,cv2.TM_CCOEFF_NORMED)
    _,best,_,loc=cv2.minMaxLoc(corr)
    lx,ly=loc
    return template,area[ly:ly+th,lx:lx+tw],float(best)

def features(r,t):
    if r is None or t is None or r.size==0 or t.size==0:return None
    if r.shape[:2]!=t.shape[:2]:t=cv2.resize(t,(r.shape[1],r.shape[0]))
    rg=normalized_gray(r);tg=normalized_gray(t)
    # Compare equalized local contrast to reduce brightness effects.
    rg=cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)).apply(rg)
    tg=cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)).apply(tg)
    win=min(7,min(rg.shape[:2]))
    if win%2==0:win-=1
    similarity=float(ssim(rg,tg,data_range=255,win_size=win)) if win>=3 else 0.
    diff=cv2.absdiff(rg,tg)
    change=float(np.mean(diff>35))
    er=cv2.Canny(rg,55,140);et=cv2.Canny(tg,55,140)
    edge_loss=float(np.count_nonzero((er>0)&(et==0))/max(1,np.count_nonzero(er)))
    edge_gain=float(np.count_nonzero((er==0)&(et>0))/max(1,np.count_nonzero(er)))
    # An absent component tends to remove strong edges and texture, but not always.
    ref_texture=float(np.std(rg));test_texture=float(np.std(tg))
    texture_loss=max(0.,(ref_texture-test_texture)/max(ref_texture,1.))
    # Score combines independent evidence. No XML used.
    score=0.38*(1-max(-1,similarity))/2 + 0.30*change + 0.20*min(edge_loss,1) + 0.12*texture_loss
    return dict(ssim=similarity,change=change,edge_loss=edge_loss,edge_gain=edge_gain,texture_loss=texture_loss,score=score,diff=diff)

def rotation_score(r,t):
    """Optional 90/180/270-degree appearance test; only meaningful for asymmetric components."""
    rg=normalized_gray(r);tg=normalized_gray(t)
    if min(rg.shape[:2])<12 or np.std(rg)<12:return None,None
    def compare(x):
        x=cv2.resize(x,(tg.shape[1],tg.shape[0]))
        return float(ssim(cv2.resize(x,(64,64)),cv2.resize(tg,(64,64)),data_range=255))
    values={0:compare(rg),90:compare(cv2.rotate(rg,cv2.ROTATE_90_CLOCKWISE)),180:compare(cv2.rotate(rg,cv2.ROTATE_180)),270:compare(cv2.rotate(rg,cv2.ROTATE_90_COUNTERCLOCKWISE))}
    angle=max(values,key=values.get)
    return angle,values[angle]-values[0]

def detect(ref,test,objects,search,threshold,rotation,rotation_gain):
    rows=[];overlay=test.copy();regions={}
    for name,meta in objects.items():
        r,t,corr=local_match(ref,test,meta['box'],search)
        f=features(r,t)
        if f is None:continue
        angle,gain=rotation_score(r,t) if rotation else (None,None)
        is_rotation=bool(rotation and angle in (90,180,270) and gain is not None and gain>=rotation_gain and f['score']>=threshold)
        is_missing=bool(f['score']>=threshold and not is_rotation)
        status='ROTATION CANDIDATE' if is_rotation else ('CHANGE / MISSING CANDIDATE' if is_missing else 'NO STRONG CHANGE')
        color=(0,165,255) if is_rotation else ((0,0,255) if is_missing else (0,200,0))
        x1,y1,x2,y2=meta['box'];cv2.rectangle(overlay,(x1,y1),(x2,y2),color,2)
        cv2.putText(overlay,name,(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
        rows.append({'Component':name,'Status':status,'Anomaly score':round(f['score'],4),'SSIM':round(f['ssim'],4),'Changed pixels':round(f['change'],4),'Edge loss':round(f['edge_loss'],4),'Texture loss':round(f['texture_loss'],4),'Local match':round(corr,3),'Rotation angle candidate':angle,'Rotation advantage':round(gain,3) if gain is not None else None})
        regions[name]=(r,t,f['diff'])
    return pd.DataFrame(rows),overlay,regions

def metrics(actual,pred,universe):
    actual=set(actual);pred=set(pred);universe=set(universe)
    tp=len(actual&pred);fp=len(pred-actual);fn=len(actual-pred);tn=len(universe-(actual|pred))
    prec=tp/(tp+fp) if tp+fp else None
    recall=tp/(tp+fn) if tp+fn else None
    f1=2*prec*recall/(prec+recall) if prec is not None and recall is not None and prec+recall else None
    return {'TP':tp,'FP':fp,'FN':fn,'TN':tn,'Precision':prec,'Recall':recall,'F1':f1}

def label(m):return 'N/A' if m is None else f'{m:.1%}'

st.title('PCB Component Inspection — Improved Classical CV Baseline')
st.caption('No training, ML, or deep learning. Reference XML supplies ROIs; test XML is used only after prediction for evaluation.')
with st.sidebar:
    root=st.text_input('Annotated dataset directory',str(DEFAULT))
paths=discover(root)
if not paths:
    st.error('No image/XML pairs found. Point to NanoNetsPCBChecker/Dataset (the uploaded ZIP contains this directory).')
    st.stop()
clean=[p for p in paths if len(annotations(p))==60 and not any(v['rotated'] for v in annotations(p).values())]
st.info(f'Found {len(paths)} annotated image/XML pairs. {len(clean)} clean-reference candidates by XML (verify visually).')
with st.sidebar:
    refs=clean if clean else paths
    reference=st.selectbox('Reference PCB',refs,format_func=lambda p:Path(p).name)
    matched=[p for p in paths if p!=reference and pair_key(p)==pair_key(reference)]
    tests=matched if matched else [p for p in paths if p!=reference]
    test=st.selectbox('Test PCB',tests,format_func=lambda p:Path(p).name)
    st.caption('Choose a matching board pair (e.g. 10.jpg → 10.10.jpg).')
    threshold=st.slider('Anomaly score threshold',0.05,0.80,0.25,0.01)
    search=st.slider('Local alignment search (pixels)',0,20,6)
    rotation=st.checkbox('Experimental 90° rotation comparison',False)
    rotation_gain=st.slider('Rotation similarity advantage',0.05,0.60,0.18,0.01,disabled=not rotation)
    run=st.button('Run Inspection',type='primary',use_container_width=True)
if pair_key(reference)!=pair_key(test):st.warning('Different board IDs selected. Component positions may not correspond; evaluation may be invalid.')
if not run:st.stop()
ref=load(reference);tst=load(test)
if ref is None or tst is None:st.error('Unable to read one of the selected images.');st.stop()
reference_objects=annotations(reference);test_objects=annotations(test)
with st.spinner('Registering PCB images and comparing component ROIs...'):
    aligned,corr,ok=register(ref,tst)
    frame,overlay,regions=detect(ref,aligned,reference_objects,search,threshold,rotation,rotation_gain)
if frame.empty:st.error('No components processed.');st.stop()
st.subheader('1. Input and alignment')
c1,c2,c3=st.columns(3)
with c1:st.image(cv2.cvtColor(ref,cv2.COLOR_BGR2RGB),caption='Reference',use_container_width=True)
with c2:st.image(cv2.cvtColor(tst,cv2.COLOR_BGR2RGB),caption='Original test',use_container_width=True)
with c3:st.image(cv2.cvtColor(aligned,cv2.COLOR_BGR2RGB),caption=f'Aligned test (ECC {corr:.3f})' if ok else 'Unaligned test — ECC failed',use_container_width=True)
if not ok:st.warning('ECC alignment failed. Predictions may be inaccurate.')
st.subheader('2. Component comparison')
st.image(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB),caption='Green: no strong change; red: changed/missing candidate; orange: rotation candidate',use_container_width=True)
st.dataframe(frame,use_container_width=True,hide_index=True)
st.download_button('Download predictions CSV',frame.to_csv(index=False).encode(),file_name=f'pcb_{Path(test).stem}.csv',mime='text/csv')
choice=st.selectbox('Inspect component ROI',frame['Component'].tolist())
a,b,d=regions[choice]
c1,c2,c3=st.columns(3)
with c1:st.image(cv2.cvtColor(a,cv2.COLOR_BGR2RGB),caption='Reference ROI',use_container_width=True)
with c2:st.image(cv2.cvtColor(b,cv2.COLOR_BGR2RGB),caption='Test ROI',use_container_width=True)
with c3:st.image(d,caption='Contrast-normalized difference',use_container_width=True,clamp=True)
st.subheader('3. XML ground-truth evaluation')
st.warning('Ground truth is read only for evaluation. Changed/missing candidates are NOT confirmed missing parts. Rotation matching is experimental. Do not report accuracy from a single pair.')
actual_missing=set(reference_objects)-set(test_objects)
actual_rotated={k for k,v in test_objects.items() if v['rotated']==1}
pred_missing=set(frame.loc[frame['Status']=='CHANGE / MISSING CANDIDATE','Component'])
pred_rot=set(frame.loc[frame['Status']=='ROTATION CANDIDATE','Component'])
for title,actual,pred in [('Missing-component candidates',actual_missing,pred_missing),('Rotation candidates',actual_rotated,pred_rot)]:
    st.markdown(f'#### {title}')
    m=metrics(actual,pred,reference_objects)
    a,b,c,d=st.columns(4)
    a.metric('Actual',len(actual));b.metric('Predicted',len(pred));c.metric('Precision',label(m['Precision']));d.metric('Recall',label(m['Recall']))
    st.caption(f"F1={label(m['F1'])} | TP={m['TP']}, FP={m['FP']}, FN={m['FN']}")
    c1,c2=st.columns(2)
    c1.write('XML ground truth: '+(', '.join(sorted(actual)) or 'None'))
    c2.write('Pixel predictions: '+(', '.join(sorted(pred)) or 'None'))
st.caption('For a valid project evaluation: use multiple matching pairs, choose parameters on development pairs, and report results on separate held-out pairs. No threshold can guarantee perfect detection.')
