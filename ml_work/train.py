import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, sys, json, time
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score
torch.set_num_threads(2); 
model_name, pfile, epochs, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
tag = sys.argv[5] if len(sys.argv)>5 else f"{model_name}_{pfile.split('.')[0]}_s{seed}"
torch.manual_seed(seed); np.random.seed(seed)
d=np.load(pfile); P=d["P"]; Y=d["y"]; Tt=d["t"]
N,T,C,R,_=P.shape
blk=Tt//60_000_000; start_in_blk=(Tt-blk*60_000_000)/1e6
role=np.where(blk%5==4,2,np.where(blk%5==3,1,0))
ok=(role==0)|(start_in_blk>1.0)           # 1 s guard at block start for val/test
tr=np.nonzero((role==0)&ok)[0]; va=np.nonzero((role==1)&ok)[0]; te=np.nonzero((role==2)&ok)[0]
print(f"N={N} train {len(tr)} val {len(va)} test {len(te)} pos% tr {Y[tr].mean():.3f} te {Y[te].mean():.3f}",flush=True)

class Spk(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x): ctx.save_for_backward(x); return (x>0).float()
    @staticmethod
    def backward(ctx,g): (x,)=ctx.saved_tensors; return g/(1+10*x.abs())**2
spk=Spk.apply
class CNN(nn.Module):
    def __init__(s):
        super().__init__(); s.c1=nn.Conv2d(2*T,16,3,padding=1); s.c2=nn.Conv2d(16,32,3,padding=1)
        s.pool=2 if R==16 else 4; f=32*(R//2 if R==16 else R//4)**2*1
        s.fc1=nn.Linear(32*(R//2)**2 if R==16 else 32*(R//4)**2,128); s.fc2=nn.Linear(128,2)
    def forward(s,x):                      # x: B,T,2,R,R counts
        x=torch.clamp(x,0,3)/3; x=x.flatten(1,2)
        x=F.relu(s.c1(x)); 
        if R==32: x=F.max_pool2d(x,2)
        x=F.max_pool2d(F.relu(s.c2(x)),2); return s.fc2(F.relu(s.fc1(x.flatten(1)))),{}
class CSNN(nn.Module):
    def __init__(s,beta=0.8):
        super().__init__(); s.beta=beta; s.c1=nn.Conv2d(2,16,3,padding=1); s.c2=nn.Conv2d(16,32,3,padding=1)
        s.fc1=nn.Linear(32*(R//2)**2 if R==16 else 32*(R//4)**2,128); s.fc2=nn.Linear(128,2)
    def forward(s,x):
        x=(x>0).float(); B=x.shape[0]; m1=m2=m3=0; out=0; r1=r2=r3=0
        for k in range(T):
            m1=s.beta*m1+s.c1(x[:,k]); s1=spk(m1-1); m1=m1-s1
            h=s1 if R==16 else F.max_pool2d(s1,2)
            m2=s.beta*m2+s.c2(h) if True else 0
            if k==0 and isinstance(m2,int): pass
            s2=spk(m2-1); m2=m2-s2; h=F.max_pool2d(s2,2)
            m3=s.beta*m3+s.fc1(h.flatten(1)); s3=spk(m3-1); m3=m3-s3
            out=out+s.fc2(s3); r1+=s1.mean().item(); r2+=s2.mean().item(); r3+=s3.mean().item()
        return out/T,{"rate_c1":r1/T,"rate_c2":r2/T,"rate_fc1":r3/T}
class FCSNN(nn.Module):
    def __init__(s,beta=0.8):
        super().__init__(); s.beta=beta; s.f1=nn.Linear(2*R*R,128); s.f2=nn.Linear(128,2)
    def forward(s,x):
        x=(x>0).float().flatten(2); m=0; out=0; r=0
        for k in range(T):
            m=s.beta*m+s.f1(x[:,k]); sp=spk(m-1); m=m-sp; out=out+s.f2(sp); r+=sp.mean().item()
        return out/T,{"rate_fc1":r/T}
M={"cnn":CNN,"csnn":CSNN,"fcsnn":FCSNN}[model_name]()
npar=sum(p.numel() for p in M.parameters()); print("params",npar,flush=True)
opt=torch.optim.Adam(M.parameters(),lr=2e-3)
Pt=torch.from_numpy(P); Yt=torch.from_numpy(Y).long()
def predict(ix,bs=512):
    M.eval(); ps=[]; rates={}
    with torch.no_grad():
        for i in range(0,len(ix),bs):
            b=ix[i:i+bs]; o,r=M(Pt[b].float()); ps.append(torch.softmax(o,1)[:,1].numpy())
            for k,v in r.items(): rates.setdefault(k,[]).append(v)
    return np.concatenate(ps),{k:float(np.mean(v)) for k,v in rates.items()}
def f1_at(y,p,th):
    pr=p>=th; tp=(pr&(y==1)).sum(); fp=(pr&(y==0)).sum(); fn=(~pr&(y==1)).sum()
    pre=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); return pre,rec,2*pre*rec/max(pre+rec,1e-9)
t0=time.time(); best=(-1,None)
for ep in range(epochs):
    M.train(); perm=np.random.permutation(tr); tl=0
    for i in range(0,len(perm),128):
        b=perm[i:i+128]; o,_=M(Pt[b].float()); loss=F.cross_entropy(o,Yt[b]); opt.zero_grad(); loss.backward(); opt.step(); tl+=loss.item()*len(b)
    pv,_=predict(va); ap=average_precision_score(Y[va],pv)
    print(f"ep{ep} loss {tl/len(perm):.4f} val AP {ap:.4f} t {time.time()-t0:.0f}s",flush=True)
    if ap>best[0]: best=(ap,{k:v.clone() for k,v in M.state_dict().items()})
M.load_state_dict(best[1])
pv,_=predict(va); pt,rates=predict(te)
ths=np.linspace(0.05,0.95,91); th=ths[np.argmax([f1_at(Y[va],pv,t_)[2] for t_ in ths])]
pre,rec,f1=f1_at(Y[te],pt,th)
res=dict(tag=tag,model=model_name,R=R,T=T,params=npar,test_n=int(len(te)),test_pos=float(Y[te].mean()),th=float(th),precision=float(pre),recall=float(rec),f1=float(f1),
 test_AP=float(average_precision_score(Y[te],pt)),test_AUC=float(roc_auc_score(Y[te],pt)),input_density=float((P[te]>0).mean()),rates=rates,train_s=time.time()-t0,seed=seed)
print(json.dumps(res),flush=True)
json.dump(res,open(f"res_{tag}.json","w")); torch.save(M.state_dict(),f"w_{tag}.pt")
