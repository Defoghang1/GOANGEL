import os, sys, tempfile, shutil, logging
ROOT=os.getcwd(); tmp=tempfile.mkdtemp(); os.chdir(tmp)
os.environ["BZZOIRO_TOKENS"]="tok_aaaaaaaaaaaa"; os.environ["GOANGEL_CACHE_DIR"]=tmp+"/cache"
sys.path[:0]=[ROOT+"/tests/stubs",ROOT]
import numpy as np, pandas as pd
logging.disable(logging.NOTSET)
import main as M, predict as P
from features import EXPECTED_FEATURE_COUNT_FULL as FULLDIM
from features import process_matches
from config import CACHE_DIR
r=np.random.default_rng(1); T=24; att=r.normal(0,.35,T); de=r.normal(0,.3,T)
rows=[]; d0=pd.Timestamp("2024-08-01",tz="UTC")
for i in range(3200):
    h,a=r.choice(T,2,replace=False); gh=r.poisson(np.exp(.25+att[h]-de[a])); ga=r.poisson(np.exp(att[a]-de[h]))
    rows.append({"id":i+1,"utcDate":(d0+pd.Timedelta(hours=i*5)).isoformat(),"status":"FINISHED",
      "homeTeam":{"id":int(h),"name":f"T{h}"},"awayTeam":{"id":int(a),"name":f"T{a}"},
      "score":{"fullTime":{"home":int(gh),"away":int(ga)},"halfTime":{"home":int(gh>0),"away":0}},
      "competition":{"code":"1","name":"L"}})
df=process_matches(rows,allowed_statuses=("FINISHED",))
print("df",len(df))
M._atomic_write_pickle(CACHE_DIR/"historical.pkl",df)
M._stage_save_features(df); sel=M._resolve_selection()
ML=("RF","HistGB","LR","Ordinal","NegBinReg"); STAT=("EloRatingModel","BayesianDynamicStrengthModel","KalmanBayesianModel","MonteCarloSimulator")
for n in ML: M._stage_train_ml_model(n)
for n in STAT: M._stage_train_pb_model(n)
M._stage_train_stacking()
M._stage_finalize_prep()
for n in ML: M._finalize_ml_one(n)
for n in STAT: M._finalize_pb_one(n)
M._finalize_ml_one("Stacking")
M._stage_finalize_agg()

def run(label):
    ml,pb,pipe,dfp,w=P.load_data_and_models()
    print(label,"-> ML",None if ml is None else list(ml),"PB",None if pb is None else len(pb),"pipe",type(pipe).__name__)
    trk,elo=P.build_tracking_state(dfp)
    m=P.predict_match("id:1","id:2","T1","T2","1",ml,pb,pipe,dfp,trk,elo,w,pd.Timestamp("2026-10-26",tz="UTC"),0.45,None)
    print(label,"-> fallback",m["fallback"],"ml",m["model_count_ml"],"pb",m["model_count_pb"],"hda",np.round(m["home_draw_away"],3))
    return m
m=run("NEUF")
assert not m["fallback"] and m["model_count_pb"]>=1, (m["model_count_ml"], m["model_count_pb"])
pipe=M._atomic_read_pickle(CACHE_DIR/"scaler.pkl")
saved=pipe.selected_idx
del pipe.__dict__["selected_idx"]
M._atomic_write_pickle(CACHE_DIR/"scaler.pkl",pipe)
import io, contextlib
ml,pb,pp,dfp,w=P.load_data_and_models()
ml={"RF":M._atomic_read_pickle(M.MODELS_DIR/"ml_RF.pkl")["model"]}
buf=io.StringIO()
with contextlib.redirect_stdout(buf): ok=P._diagnose_models(ml,pb,pp)
print(buf.getvalue()); assert ok is False and "❌" in buf.getvalue()
trk,elo=P.build_tracking_state(dfp)
m=P.predict_match("id:1","id:2","T1","T2","1",ml,pb,pp,dfp,trk,elo,w,pd.Timestamp("2026-10-26",tz="UTC"),0.45,None)
print("B fallback",m["fallback"],"| raison:",m.get("fallback_reason")); assert m["model_count_ml"]==0

shutil.rmtree(M.MODELS_DIR); M.MODELS_DIR.mkdir(parents=True)
shutil.rmtree(Path:=__import__("pathlib").Path(CACHE_DIR)/"stages"/"finalize",ignore_errors=True)
orig=M._resolve_selection
M._resolve_selection=lambda: {**orig(),"selected_idx":list(range(FULLDIM)),"selection_id":"legacy","n_features":FULLDIM}
ML=("RF","HistGB","LR","Ordinal","NegBinReg"); STAT=("EloRatingModel","BayesianDynamicStrengthModel","KalmanBayesianModel","MonteCarloSimulator")
for n in ML: M._stage_train_ml_model(n)
for n in STAT: M._stage_train_pb_model(n)
M._stage_train_stacking()
M._stage_finalize_prep()
for n in ML: M._finalize_ml_one(n)
for n in STAT: M._finalize_pb_one(n)
M._finalize_ml_one("Stacking")
M._stage_finalize_agg()
pipe=M._atomic_read_pickle(CACHE_DIR/"scaler.pkl"); del pipe.__dict__["selected_idx"]
M._atomic_write_pickle(CACHE_DIR/"scaler.pkl",pipe)
m=run("A-ANCIEN-COHERENT")
assert not m["fallback"] and m["model_count_ml"]+m["model_count_pb"]>=2
import json
dp=M._stage_done_path("train_xgb"); dp.write_text(json.dumps({"stage":"train_xgb","files":{},"feature_schema":172}))
assert not M._is_stage_done("train_xgb")
assert M._purge_stale_markers()>=1 and not dp.exists()
print("FULL CHAIN DONE")
