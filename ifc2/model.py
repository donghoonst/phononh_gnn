"""Independent equivariant pair network, using e3nn primitives (no vendored model)."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from e3nn import o3
from .physics import scatter, reconstruct, constrain_pairs


@dataclass
class ModelConfig:
    width: int = 4
    layers: int = 2
    lmax: int = 2
    num_rbf: int = 8
    conv_cutoff: float = 4.5
    pair_cutoff: float = 5.5
    correlation: int = 2
    attention: bool = True
    attention_heads: int = 2
    aggregation: str = "sum_attention"
    persistent_pair: bool = True
    enhanced_fusion: bool = False
    physics_residual: bool = False
    independent_heads: bool = False

    def __post_init__(self):
        if self.lmax < 1 or self.width<1 or self.layers<1: raise ValueError("Invalid model dimensions")
        if self.correlation not in (1,2,3): raise ValueError("correlation must be 1, 2 or 3")
        if self.aggregation not in ("sum_attention","attention_only"): raise ValueError("Invalid aggregation")
        if self.attention_heads<1 or min(self.conv_cutoff,self.pair_cutoff)<=0: raise ValueError("Invalid heads/cutoff")


def geometry(v,cutoff,num_rbf,lmax):
    distance=v.norm(dim=-1)
    x=distance/cutoff
    envelope=torch.where(x<1,(1-x).clamp_min(0)**3*(1+3*x+6*x*x),torch.zeros_like(x))
    k=torch.arange(1,num_rbf+1,device=v.device,dtype=v.dtype)
    radial=torch.sinc(x[:,None]*k)*envelope[:,None]
    sh=o3.spherical_harmonics(list(range(lmax+1)),v,normalize=True,normalization="component")
    return radial,sh,envelope


class InvariantGate(nn.Module):
    def __init__(self,irreps,width):
        super().__init__(); self.irreps=irreps; self.width=width
        self.gate=nn.Linear(width,irreps.num_irreps-width)

    def forward(self,x):
        weights=torch.sigmoid(self.gate(x[:,:self.width])); pieces=[torch.nn.functional.silu(x[:,:self.width])]
        start=0
        for (mul,ir),sl in list(zip(self.irreps,self.irreps.slices()))[1:]:
            pieces.append((x[:,sl].reshape(-1,mul,ir.dim)*weights[:,start:start+mul,None]).flatten(1))
            start+=mul
        return torch.cat(pieces,-1)


class RadialTP(nn.Module):
    def __init__(self,ir,sh,out,rbf,width):
        super().__init__()
        self.tp=o3.FullyConnectedTensorProduct(ir,sh,out,shared_weights=False)
        self.radial=nn.Sequential(nn.Linear(rbf,2*width),nn.SiLU(),nn.Linear(2*width,self.tp.weight_numel))

    def forward(self,x,sh,radial,envelope):
        return self.tp(x,sh,self.radial(radial))*envelope[:,None]


class Interaction(nn.Module):
    def __init__(self,ir,c):
        super().__init__();self.c=c;self.ir=ir
        self.message=RadialTP(ir,o3.Irreps.spherical_harmonics(c.lmax),ir,c.num_rbf,c.width)
        self.mix=o3.Linear(ir,ir)
        self.gate=InvariantGate(ir,c.width)
        if c.correlation>1:
            # ACE-inspired recursive density products, not MACE's complete symmetric basis.
            self.products=nn.ModuleList([o3.FullyConnectedTensorProduct(ir,ir,ir) for _ in range(c.correlation-1)])
        if c.attention:
            self.score=nn.Sequential(nn.Linear(2*c.width+c.num_rbf,c.width),nn.SiLU(),nn.Linear(c.width,c.attention_heads))
            self.values=nn.ModuleList([o3.Linear(ir,ir) for _ in range(c.attention_heads)])
            self.att_mix=o3.Linear(ir*c.attention_heads,ir)

    def forward(self,h,index,radial,sh,envelope):
        i,j=index;n=len(h)
        # Remove outside-cutoff entries BEFORE attention normalization.
        active=envelope>0;i,j=i[active],j[active]
        radial,sh,envelope=radial[active],sh[active],envelope[active]
        m=self.message(h[j],sh,radial,envelope)
        density=scatter(m,i,n)/math.sqrt(12.0)
        update=self.mix(density)
        if self.c.attention:
            scores=self.score(torch.cat([h[i,:self.c.width],h[j,:self.c.width],radial],-1))
            attended=[]
            for head,value_layer in enumerate(self.values):
                # Local segment softmax; small reference implementation, no torch_scatter dependency.
                rows=[];values=value_layer(m)
                for atom in range(n):
                    mask=i==atom
                    if mask.any():
                        logits=scores[mask,head]+torch.log(envelope[mask])
                        rows.append((logits.softmax(0)[:,None]*values[mask]).sum(0))
                    else: rows.append(h.new_zeros(self.ir.dim))
                attended.append(torch.stack(rows))
            attention=self.att_mix(torch.cat(attended,-1))
            update=attention if self.c.aggregation=="attention_only" else update+attention
        if self.c.correlation>1:
            product=density
            for tp in self.products:
                product=tp(product,density)
                update=update+product
        return h+self.gate(update)/math.sqrt(self.c.layers)


class PairBackbone(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c
        # Include both polar 1o and axial 1e. lmax controls geometric SH order.
        self.ir=o3.Irreps([(c.width,(0,1))]+[(c.width,(l,p)) for l in range(1,c.lmax+1) for p in (-1,1)])
        self.embedding=nn.Embedding(119,c.width)
        self.embed=o3.Linear(f"{c.width}x0e",self.ir)
        self.layers=nn.ModuleList([Interaction(self.ir,c) for _ in range(c.layers)])
        sh=o3.Irreps.spherical_harmonics(c.lmax)
        self.pair_init=RadialTP(self.ir+self.ir,sh,self.ir,c.num_rbf,c.width)
        self.pair_updates=nn.ModuleList([RadialTP(self.ir+self.ir,sh,self.ir,c.num_rbf,c.width) for _ in range(c.layers if c.persistent_pair else 1)])
        self.pair_skips=nn.ModuleList([o3.Linear(self.ir,self.ir) for _ in self.pair_updates])
        self.pair_gates=nn.ModuleList([InvariantGate(self.ir,c.width) for _ in self.pair_updates])
        if c.enhanced_fusion:
            self.pair_context=o3.Linear(self.ir+self.ir,self.ir)
            self.pair_products=nn.ModuleList([o3.FullyConnectedTensorProduct(self.ir,self.ir,self.ir) for _ in self.pair_updates])
        self.readout=o3.Linear(self.ir,"0e+1e+2e")
        if c.physics_residual:
            self.stiffness=nn.Sequential(nn.Linear(2*c.width+c.num_rbf,c.width),nn.SiLU(),nn.Linear(c.width,2))

    def forward(self,s):
        c=self.c;h=self.embed(self.embedding(s["z"]))
        cg=geometry(s["conv_vectors"],c.conv_cutoff,c.num_rbf,c.lmax)
        pg=geometry(s["pair_vectors"],c.pair_cutoff,c.num_rbf,c.lmax)
        owners=s["image_owner"];i,j=s["pair_index"][:,owners]
        pair=self.pair_init(torch.cat([h[i],h[j]],-1),pg[1],pg[0],pg[2])
        for layer,block in enumerate(self.layers):
            h=block(h,s["conv_index"],*cg)
            if c.persistent_pair:
                update=self.pair_updates[layer](torch.cat([h[i],h[j]],-1),pg[1],pg[0],pg[2])
                if c.enhanced_fusion:
                    update=update+self.pair_products[layer](pair,self.pair_context(torch.cat([h[i],h[j]],-1)))
                pair=pair+self.pair_gates[layer](self.pair_skips[layer](pair)+update)/math.sqrt(c.layers)
        if not c.persistent_pair:
            pair=self.pair_gates[0](self.pair_skips[0](pair)+self.pair_updates[0](torch.cat([h[i],h[j]],-1),pg[1],pg[0],pg[2]))
        images=reconstruct(self.readout(pair))
        if c.physics_residual:
            kl,kt=self.stiffness(torch.cat([h[i,:c.width],h[j,:c.width],pg[0]],-1)).unbind(-1)
            unit=torch.nn.functional.normalize(s["pair_vectors"],dim=-1)
            outer=unit[:,:,None]*unit[:,None,:]
            images=images+(kl[:,None,None]*outer+kt[:,None,None]*(torch.eye(3,device=h.device,dtype=h.dtype)-outer))*pg[2][:,None,None]
        return images


class IFCModel(nn.Module):
    def __init__(self,config=None):
        super().__init__(); self.config=config or ModelConfig()
        self.backbone=PairBackbone(self.config)
        if self.config.independent_heads: self.antisymmetric_backbone=PairBackbone(self.config)

    def forward(self,s,return_images=False):
        raw=self.backbone(s)
        if self.config.independent_heads:
            other=self.antisymmetric_backbone(s)
            raw=(raw+raw.transpose(-1,-2))/2+(other-other.transpose(-1,-2))/2
        raw=(raw+raw[s["image_reverse"]].transpose(-1,-2))/2
        owner=s["image_owner"];count=torch.bincount(owner,minlength=s["pair_index"].shape[1]).to(raw)
        grouped=scatter(raw,owner,len(count))
        pairs=constrain_pairs(grouped,s)
        if return_images:
            # Minimum equal correction within each alias group; individual values remain underdetermined by compact labels.
            images=raw+(pairs-grouped)[owner]/count[owner,None,None]
            return pairs,images
        return pairs
