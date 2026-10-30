import os, sys, tempfile, shutil
ROOT=os.getcwd(); tmp = tempfile.mkdtemp(); os.chdir(tmp)
os.environ["BZZOIRO_TOKENS"]="tok_aaaaaaaaaaaa"; os.environ["GOANGEL_CACHE_DIR"]=tmp+"/cache"
sys.path[:0]=[ROOT+"/tests/stubs",ROOT]
import numpy as np
import main as M
from features import EXPECTED_FEATURE_COUNT, EXPECTED_FEATURE_COUNT_FULL
from config import CACHE_DIR
print("CACHE", CACHE_DIR)
r=np.random.default_rng(3); n=7000; p=EXPECTED_FEATURE_COUNT
X=r.normal(size=(n,p)); w=np.zeros(p); w[:25]=r.normal(0,.5,25)
z=X@w; L=np.column_stack([z+.2,0*z,-z]); Pm=np.exp(L); Pm/=Pm.sum(1,keepdims=True)
y=np.array([r.choice(3,p=q) for q in Pm]); X[r.random(X.shape)<.1]=np.nan
hk=[f"id:{i%40}" for i in range(n)]; ak=[f"id:{(i*7+3)%40}" for i in range(n)]
payload=dict(X=X,y=y,home_keys=hk,away_keys=ak,gh_all=[1]*n,ga_all=[1]*n,dates=[None]*n)
M._atomic_write_pickle(M._FEATURES_FILE,payload)
sel=M._resolve_selection()
print("selected",len(sel["selected_idx"]),"id",sel["selection_id"])
assert len(sel["selected_idx"])==141
for name in ("RF","HistGB"):
    M._stage_train_ml_model(name); M._finalize_ml_one(name)
M._stage_train_ml_model("RF")
sp=M._build_splits()
assert sp["sc_train"].shape[1]==len(sel["selected_idx"])==sp["raw_train"].shape[1]
pl=M._atomic_read_pickle(M.MODELS_DIR/"ml_RF.pkl")
assert pl["selection_id"]==sel["selection_id"] and pl["prep"].n_features_in_==EXPECTED_FEATURE_COUNT_FULL
try:
    M._stage_finalize_prep(); M._stage_finalize_agg()
except RuntimeError as e: print("agg:",e)
shutil.rmtree(tmp)
print("E2E OK")
