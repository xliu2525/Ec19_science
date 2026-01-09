#@title import libraries
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from colabdesign import clear_mem, mk_af_model
from colabdesign.shared.utils import copy_dict

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from common import memory
from log import dlog

@memory.cache(ignore=["af"])
def score_seq(af, multi_seq, model_name, recycles, rm_ic=False):
    af_score = af.predict(seq=multi_seq, model_name=model_name, recycles=recycles, rm_ic=rm_ic)
    return af_score

@memory.cache(ignore=["af"])
def score_seq_res(af, multi_seq, model_name, recycles, rm_ic=False):
    af_score = af.predict(seq=multi_seq, model_name=model_name, recycles=recycles, rm_ic=rm_ic)
    af_score['res_dgram_cce'] = af.model.aux['res_dgram_cce']
    return af_score

def score_seqs(af, seqs, initial_seq, wt_multi_seq, model_names, recycles, per_res=False):
    seq_af_scores = {}
    unique_seqs = set(seqs)
    for i, seq in enumerate(unique_seqs):
        print(f"Scoring {i}/{len(unique_seqs)-1}")
        assert(wt_multi_seq.find(initial_seq) != -1)
        multi_seq = wt_multi_seq.replace(initial_seq, seq)

        dlog(len(multi_seq), len(initial_seq), len(seq), len(wt_multi_seq))
        assert(seq in multi_seq)
        af_model_scores = []
        for model_name in model_names:
            print(f"Scoring with model {model_name}, per_res={per_res}")
            if per_res:
              af_score = score_seq_res(af, multi_seq, model_name, recycles)
            else:
              af_score = score_seq(af, multi_seq, model_name, recycles)            
            af_model_scores.append(af_score)
        
        seq_af_scores[seq] = af_model_scores
    return seq_af_scores

def tmscore(x,y):
  pid = os.getpid()
  # save to dumpy pdb files
  for n,z in enumerate([x,y]): 
    out = open(f"{n}_{pid}.pdb","w")
    for k,c in enumerate(z):
      out.write("ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n" 
                  % (k+1,"CA","ALA","A",k+1,c[0],c[1],c[2],1,0))
    out.close()
  # pass to TMscore
  output = os.popen(f'./TMscore 0_{pid}.pdb 1_{pid}.pdb')

  # parse outputs
  parse_float = lambda x: float(x.split("=")[1].split()[0])
  o = {}
  for line in output:
    line = line.rstrip()
    if line.startswith("RMSD"): o["rms"] = parse_float(line)
    if line.startswith("TM-score"): o["tms"] = parse_float(line)
    if line.startswith("GDT-TS-score"): o["gdt"] = parse_float(line)
  return o
  
def plot_me(scores, x="tm_i", y="composite", 
            title=None, diag=False, scale_axis=True, dpi=100, **kwargs):
  def rescale(a,amin=None,amax=None):  
    a = np.copy(a)
    if amin is None: amin = a.min()
    if amax is None: amax = a.max()
    a[a < amin] = amin
    a[a > amax] = amax
    return (a - amin)/(amax - amin)

  plt.figure(figsize=(5,5), dpi=dpi)
  if title is not None: plt.title(title)
  x_vals = np.array([k[x] for k in scores])
  y_vals = np.array([k[y] for k in scores])
  c = rescale(np.array([k["plddt"] for k in scores]),0.5,0.9)
  plt.scatter(x_vals, y_vals, c=c*0.75, s=5, vmin=0, vmax=1, cmap="gist_rainbow",
              **kwargs)
  if diag:
    plt.plot([0,1],[0,1],color="black")
  
  labels = {"tm_i":"TMscore of Input",
            "tm_o":"TMscore of Output",
            "tm_io":"TMscore between Input and Output",
            "ptm":"Predicted TMscore (pTM)",
            "i_ptm":"Predicted interface TMscore (ipTM)",
            "plddt":"Predicted LDDT (pLDDT)",
            "composite":"Composite"}

  plt.xlabel(labels.get(x,x));  plt.ylabel(labels.get(y,y))
  if scale_axis:
    if x in labels: plt.xlim(-0.1, 1.1)
    if y in labels: plt.ylim(-0.1, 1.1)
  
  print(spearmanr(x_vals,y_vals).correlation)

class af2rank:
  def __init__(self, pdb, chain=None, model_name="model_1_ptm", model_names=None):
    self.args = {"pdb":pdb, "chain":chain,
                 "use_multimer":("multimer" in model_name),
                 "model_name":model_name,
                 "model_names":model_names}
    self.reset()

  def reset(self):
    self.model = mk_af_model(protocol="fixbb",
                             use_templates=True,
                             use_multimer=self.args["use_multimer"],
                             #use_alphafold=True,
                             debug=False,
                             model_names=self.args["model_names"])
    
    self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
    self.model.set_seq(mode="wildtype")
    self.wt_batch = copy_dict(self.model._inputs["batch"])
    self.wt = self.model._wt_aatype

  def set_pdb(self, pdb, chain=None):
    if chain is None: chain = self.args["chain"]
    self.model.prep_inputs(pdb, chain=chain)
    self.model.set_seq(mode="wildtype")
    self.wt = self.model._wt_aatype

  def set_seq(self, seq):
    self.model.set_seq(seq=seq)
    self.wt = self.model._params["seq"][0].argmax(-1)

  def _get_score(self):
    score = copy_dict(self.model.aux["log"])

    score["plddt"] = score["plddt"]
    score["pae"] = 31.0 * score["pae"]
    score["rmsd_io"] = score.pop("rmsd",None)

    i_xyz = self.model._inputs["batch"]["all_atom_positions"][:,1]
    o_xyz = np.array(self.model.aux["atom_positions"][:,1])

    # TMscore to input/output
    if hasattr(self,"wt_batch"):
      n_xyz = self.wt_batch["all_atom_positions"][:,1]
      score["tm_i"] = tmscore(n_xyz,i_xyz)["tms"]
      score["tm_o"] = tmscore(n_xyz,o_xyz)["tms"]

    # TMscore between input and output
    score["tm_io"] = tmscore(i_xyz,o_xyz)["tms"]

    # composite score
    score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
    return score
  
  def predict(self, pdb=None, seq=None, chain=None, 
              input_template=True, model_name=None,
              rm_seq=True, rm_sc=True, rm_ic=False,
              recycles=1, iterations=1,
              output_pdb=None, extras=None, verbose=True):
    
    if model_name is not None:
      self.args["model_name"] = model_name
      if "multimer" in model_name: 
        if not self.args["use_multimer"]:
          self.args["use_multimer"] = True
          self.reset()
      else:
        if self.args["use_multimer"]:
          self.args["use_multimer"] = False
          self.reset()
  
    if pdb is not None: self.set_pdb(pdb, chain)
    if seq is not None: self.set_seq(seq)

    # set template sequence
    self.model._inputs["batch"]["aatype"] = self.wt

    # set other options
    self.model.set_opt(
        template=dict(rm_ic=rm_ic),
        dropout=not input_template,
        num_recycles=recycles)
    self.model._inputs["rm_template_sc"][:] = rm_sc
    self.model._inputs["rm_template_seq"][:] = rm_seq
    # zero out 'sym_id', 'asym_id', 'entity_id'
    self.model._inputs['sym_id'] = np.zeros_like(self.model._inputs['sym_id'])
    self.model._inputs['asym_id'] = np.zeros_like(self.model._inputs['asym_id'])
    self.model._inputs['entity_id'] = np.zeros_like(self.model._inputs['entity_id'])
    
    # "manual" recycles using templates
    ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()

    self.model._inputs["all_atom_positions"] = ini_atoms
    for i in range(iterations):
      self.model.predict(models=self.args["model_name"], verbose=False)
      if i < iterations - 1:
        self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
      else:
        self.model._inputs["batch"]["all_atom_positions"] = ini_atoms
    
    score = self._get_score()
    if extras is not None:
      score.update(extras)

    if output_pdb is not None:
      self.model.save_pdb(output_pdb)
    
    if verbose:
      print_list = ["tm_i","tm_o","tm_io","composite","ptm","i_ptm","plddt","fitness","id"]
      print_score = lambda k: f"{k} {score[k]:.4f}" if isinstance(score[k],float) else f"{k} {score[k]}"
      print(*[print_score(k) for k in print_list if k in score])

    return score
     