import numpy as np,json,sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
pf=sys.argv[1]; d=np.load(pf); P=d["P"]; Y=d["y"]; Tt=d["t"]
blk=Tt//60_000_000; s=(Tt-blk*60_000_000)/1e6; role=np.where(blk%5==4,2,np.where(blk%5==3,1,0)); ok=(role==0)|(s>1)
tr=np.nonzero((role==0)&ok)[0];va=np.nonzero((role==1)&ok)[0];te=np.nonzero((role==2)&ok)[0]
N,T,C,R,_=P.shape
F=np.c_[(P>0).sum((1,2,3,4)),P.sum((1,2,3,4)),(P.sum((1,3,4))>0).sum(1),P.sum((2,3,4)).reshape(N,-1)]  # count, spikes, active bins, per-bin counts
def f1b(y,p):
    best=(0,0,0,0)
    for th in np.linspace(0.05,0.95,91):
        pr=p>=th; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=(~pr&(y==1)).sum()
        pre=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f=2*pre*rec/max(pre+rec,1e-9)
        if f>best[0]: best=(f,pre,rec,th)
    return best
out={}
# count-only threshold baseline
c=F[:,0].astype(float); 
from sklearn.linear_model import LogisticRegression as LR
m=LR(max_iter=500).fit(np.log1p(c[tr,None]),Y[tr]); pv=m.predict_proba(np.log1p(c[va,None]))[:,1]; pt=m.predict_proba(np.log1p(c[te,None]))[:,1]
f,pre,rec,th=f1b(Y[va],pv); 
def at(y,p,th):
    pr=p>=th; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=(~pr&(y==1)).sum(); a=tp/max(tp+fp,1); b=tp/max(tp+fn,1); return a,b,2*a*b/max(a+b,1e-9)
a,b,ff=at(Y[te],pt,th); out["count_only"]=dict(precision=a,recall=b,f1=ff,AP=average_precision_score(Y[te],pt),AUC=roc_auc_score(Y[te],pt))
g=HistGradientBoostingClassifier(max_iter=200).fit(F[tr],Y[tr]); pv=g.predict_proba(F[va])[:,1]; pt=g.predict_proba(F[te])[:,1]
f,pre,rec,th=f1b(Y[va],pv); a,b,ff=at(Y[te],pt,th); out["gbdt_count_features"]=dict(precision=a,recall=b,f1=ff,AP=average_precision_score(Y[te],pt),AUC=roc_auc_score(Y[te],pt))
print(pf,"test prevalence",Y[te].mean(),"n",len(te)); print(json.dumps(out,indent=1,default=float)); json.dump(out,open(f"base_{pf.split('.')[0]}.json","w"),default=float)
