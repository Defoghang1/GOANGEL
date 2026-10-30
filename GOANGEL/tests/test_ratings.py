import os, sys, copy
os.environ["BZZOIRO_TOKENS"]="tok_aaaaaaaaaaaa"
sys.path[:0]=["tests/stubs","."]
import numpy as np, pandas as pd
from scipy.stats import poisson
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import features as F

def world(n=9000, comps=4, T=20, seed=5):
    r=np.random.default_rng(seed)
    att=r.normal(0,.30,(comps,T)); de=r.normal(0,.25,(comps,T)); rows=[]; d0=pd.Timestamp("2022-08-01",tz="UTC")
    orac=[]
    for i in range(n):
        c=i%comps; h,a=r.choice(T,2,replace=False)
        att[c]+=r.normal(0,.01,T); de[c]+=r.normal(0,.01,T)
        lh=np.exp(.28+att[c,h]-de[c,a]); la=np.exp(att[c,a]-de[c,h])
        gh,ga=int(r.poisson(lh)),int(r.poisson(la))
        k=np.arange(12); m=np.outer(poisson.pmf(k,lh),poisson.pmf(k,la)); m/=m.sum()
        orac.append([np.tril(m,-1).sum(),np.trace(m),np.triu(m,1).sum()])
        rows.append({"id":i+1,"utcDate":(d0+pd.Timedelta(minutes=int(i*198))).isoformat(),"status":"FINISHED",
          "homeTeam":{"id":int(c*100+h),"name":f"C{c}T{h}"},"awayTeam":{"id":int(c*100+a),"name":f"C{c}T{a}"},
          "score":{"fullTime":{"home":gh,"away":ga},"halfTime":{"home":0,"away":0}},"competition":{"code":str(c+1),"name":"L"}})
    return rows,np.array(orac)

def build(rows):
    df=F.process_matches(rows,allowed_statuses=("FINISHED",))
    X,y,*_=F.create_features_with_context(df,strict_competition=True)
    return df,F._compute_diff_features(X),y

def test_no_leakage_result_of_match_does_not_change_its_own_features():
    rows,_=world(1500)
    _,X1,_=build(rows)
    r2=copy.deepcopy(rows); r2[-1]["score"]["fullTime"]["home"]+=4
    _,X2,_=build(r2)
    a,b=X1[-1],X2[-1]
    assert np.allclose(a,b,equal_nan=True), "FUITE : les features du match dépendent de son résultat"
    r3=copy.deepcopy(rows); r3[700]["score"]["fullTime"]["away"]+=3
    _,X3,_=build(r3)
    assert np.allclose(X1[:701],X3[:701],equal_nan=True)
    assert not np.allclose(X1[701:],X3[701:],equal_nan=True)

def test_new_features_lower_logloss_vs_oracle():
    rows,orac=world(9000)
    df,X,y=build(rows); n=len(X)
    ct=int(n*.65); cv=int(n*.75); ce=int(n*.85)
    names=F.FEATURE_NAMES_FULL
    new_cols=[i for i,nm in enumerate(names) if any(k in nm for k in ("_pr_","pr_exp","pr_p_","elo_g","ewm","diff_pr","diff_elo_g","diff_ewm"))]
    old_cols=[i for i in range(X.shape[1]) if i not in new_cols]
    def fit_eval(cols):
        m=HistGradientBoostingClassifier(max_iter=250,learning_rate=0.03,max_depth=4,min_samples_leaf=80,
            l2_regularization=10,early_stopping=False,random_state=1)
        m.fit(X[:ct][:,cols],y[:ct])
        lls=[log_loss(y[ct:cv],p,labels=[0,1,2]) for p in m.staged_predict_proba(X[ct:cv][:,cols])]
        k=int(np.argmin(lls))+1
        m=HistGradientBoostingClassifier(max_iter=k,learning_rate=0.03,max_depth=4,min_samples_leaf=80,
            l2_regularization=10,early_stopping=False,random_state=1).fit(X[:ct][:,cols],y[:ct])
        return log_loss(y[ce:],m.predict_proba(X[ce:][:,cols]),labels=[0,1,2])
    ll_old=fit_eval(old_cols); ll_new=fit_eval(list(range(X.shape[1])))
    prior=np.array([(y[:ct]==k).mean() for k in range(3)])
    ll_const=log_loss(y[ce:],np.tile(prior,(n-ce,1)),labels=[0,1,2])
    ll_oracle=log_loss(y[ce:],np.clip(orac[:n][ce:],1e-6,1),labels=[0,1,2])
    print(f"TEST LogLoss  constant={ll_const:.4f}  anciennes features={ll_old:.4f}  +nouvelles={ll_new:.4f}  oracle(vraies forces)={ll_oracle:.4f}")
    gap_closed=(ll_old-ll_new)/(ll_old-ll_oracle)
    print(f"écart à l'oracle comblé : {gap_closed*100:.0f} %")
    assert ll_new < ll_old - 0.005
