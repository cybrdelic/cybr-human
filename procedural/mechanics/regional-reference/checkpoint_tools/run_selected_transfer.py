#!/usr/bin/env python3
"""Execute exact original transfer expressions for an explicitly selected state subset.
Only the outer state-name loop is changed in memory; original source is unmodified.
This is numerical transfer work, not a safe validation or import entry point.
"""
from pathlib import Path
import argparse,ast,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'human-preferred-v53-mechanics-20261011/src/transfer_cheek_states_v4.py'
SOURCE_SHA='7283232bf29f407ffdfd7971280fe6d4a0e6fa4e85b51cdfb834944baa76b788'
ORIGINAL_STATES=['neutral-0','right-02','right-08','right-16','released-0']
ALLOWED_STATES=['neutral-0','right-02','released-0']

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--states',nargs='+',default=['neutral-0','right-02'],choices=ALLOWED_STATES);args=a.parse_args()
    if len(set(args.states))!=len(args.states):raise RuntimeError('Repeated state')
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=SOURCE_SHA:raise RuntimeError('Original transfer source hash mismatch')
    module=ast.parse(SOURCE.read_text(),filename=str(SOURCE));matches=[]
    for n in module.body:
        if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='name':
            try:value=ast.literal_eval(n.iter)
            except (ValueError,TypeError):continue
            if value==ORIGINAL_STATES:matches.append(n)
    if len(matches)!=1:raise RuntimeError('Expected exactly one original five-state outer loop')
    matches[0].iter=ast.List(elts=[ast.Constant(v) for v in args.states],ctx=ast.Load());ast.fix_missing_locations(module)
    # Original paths, equations, float operations, outputs and gates are unchanged.
    exec(compile(module,str(SOURCE),'exec'),{'__file__':str(SOURCE),'__name__':'__main__'})
    out=ROOT/'human-preferred-v53-mechanics-20261011/cheek-skin-states-v4'
    receipt={'source_path':str(SOURCE.relative_to(ROOT)),'source_sha256':SOURCE_SHA,'original_source_modified':False,'only_change':'Outer state-name loop literal replaced in memory by requested subset; all original numerical expressions and paths unchanged.','selected_states':args.states}
    (out/'SUBSET_TRANSFER_EXECUTION.json').write_text(json.dumps(receipt,indent=2)+'\n')

if __name__=='__main__':main()
