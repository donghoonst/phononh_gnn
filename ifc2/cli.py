import argparse,json
from pathlib import Path
import torch,yaml
from .model import ModelConfig


def main():
    torch.set_default_dtype(torch.float64);torch.set_num_threads(1)
    parser=argparse.ArgumentParser(description="Equivariant direct IFC2 research prototype")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("synthetic");p.add_argument("--out",required=True);p.add_argument("--count",type=int,default=6)
    p=sub.add_parser("audit");p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--conv-cutoff",type=float,default=4.5);p.add_argument("--pair-cutoff",type=float,default=5.5)
    p=sub.add_parser("train");p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--config");p.add_argument("--epochs",type=int,default=10);p.add_argument("--reciprocal-epochs",type=int,default=0)
    p.add_argument("--seed",type=int,default=17);p.add_argument("--device",default="cpu");p.add_argument("--split")
    p=sub.add_parser("ablate");p.add_argument("--data",required=True);p.add_argument("--out",required=True)
    p.add_argument("--epochs",type=int,default=5);p.add_argument("--width",type=int,default=3);p.add_argument("--seed",type=int,default=17)
    for command in ["predict","compare"]:
        p=sub.add_parser(command);p.add_argument("--checkpoint",required=True);p.add_argument("--material",required=True);p.add_argument("--out",required=True)
        if command=="compare":p.add_argument("--nac",action="store_true");p.add_argument("--qpoints")
        else:p.add_argument("--structure",action="store_true");p.add_argument("--supercell",type=int,nargs=3,default=[2,2,2])
    args=parser.parse_args()
    if args.command=="synthetic":
        from .synthetic import make_dataset
        make_dataset(args.out,args.count);print("Created explicitly synthetic fixtures")
    elif args.command=="audit":
        from .audit import audit_dataset
        print(json.dumps(audit_dataset(args.data,args.out,args.conv_cutoff,args.pair_cutoff),indent=2))
    elif args.command=="train":
        from .training import train
        cfg=ModelConfig(**(yaml.safe_load(Path(args.config).read_text()) if args.config else {}))
        split=json.loads(Path(args.split).read_text()) if args.split else None
        print(json.dumps(train(args.data,args.out,cfg,args.epochs,args.seed,device=args.device,reciprocal_epochs=args.reciprocal_epochs,split=split),indent=2))
    elif args.command=="ablate":
        from .training import ablations
        ablations(args.data,args.out,args.epochs,args.width,args.seed)
    elif args.command=="predict":
        from .inference import predict
        print(json.dumps(predict(args.checkpoint,args.material,args.out,args.structure,args.supercell),indent=2))
    else:
        from .inference import compare
        print(json.dumps(compare(args.checkpoint,args.material,args.out,args.nac,args.qpoints),indent=2))


if __name__=="__main__":main()
