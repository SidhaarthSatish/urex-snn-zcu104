import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json, re
src=open("train.py").read(); a=src.index("class Spk"); b=src.index("M={")
T,R=10,16
exec(src[a:b])
d=np.load("p16.npz"); P=torch.from_numpy(d["P"]); Tt=d["t"]; blk=Tt//60_000_000; s=(Tt-blk*60_000_000)/1e6
te=np.random.default_rng(0).permutation(np.nonzero((blk%5==4)&(s>1))[0])[:4000]
out={}
# CSNN
M=CSNN(); M.load_state_dict(torch.load("w_csnn_p16_s0.pt")); M.eval()
tot=dict(in_syn=0,c1_syn=0,c2_syn=0,c3_syn=0,upd=0); n=0
with torch.no_grad():
    for i in range(0,len(te),500):
        x=(P[te[i:i+500]].float()>0).float(); B=x.shape[0]; m1=m2=m3=0
        for k in range(T):
            m1=M.beta*m1+M.c1(x[:,k]); s1=spk(m1-1); m1=m1-s1
            m2=M.beta*m2+M.c2(s1); s2=spk(m2-1); m2=m2-s2; h=F.max_pool2d(s2,2)
            m3=M.beta*m3+M.fc1(h.flatten(1)); s3=spk(m3-1); m3=m3-s3
            tot["in_syn"]+=x[:,k].sum().item()*16*9; tot["c1_syn"]+=s1.sum().item()*32*9
            tot["c2_syn"]+=h.sum().item()*128; tot["c3_syn"]+=s3.sum().item()*2
        n+=B
syn={k:v/n for k,v in tot.items() if k!="upd"}; syn["total_synaptic_ops"]=sum(syn.values())
syn["neuron_state_updates"]=(16*256+32*256+128)*T
syn["dense_equivalent_MACs"]=1515904*T
out["csnn"]=syn
# FC SNN
M=FCSNN(); M.load_state_dict(torch.load("w_fcsnn_p16_s0.pt")); M.eval(); tin=ts=0; n=0
with torch.no_grad():
    for i in range(0,len(te),500):
        x=(P[te[i:i+500]].float()>0).float().flatten(2); B=x.shape[0]; m=0
        for k in range(T):
            m=M.beta*m+M.f1(x[:,k]); sp=spk(m-1); m=m-sp
            tin+=x[:,k].sum().item()*128; ts+=sp.sum().item()*2
        n+=B
out["fcsnn"]=dict(in_syn=tin/n,out_syn=ts/n,total_synaptic_ops=(tin+ts)/n,neuron_state_updates=(128+2)*T,dense_equivalent_MACs=(512*128+128*2)*T)
out["cnn_MACs"]=2179456
print(json.dumps(out,indent=1)); json.dump(out,open("opcount.json","w"),indent=1)
