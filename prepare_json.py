#!/proj/wallner-b/users/x_bjowa/afsample3/run.sh /proj/wallner-b/users/x_bjowa/apps/images/af3-dev.sif python 


#/usr/bin/env python

from absl import flags,app
from pathlib import Path
import json
import random
import sys
# Input and output paths.
flags.DEFINE_string(
    'fasta',
    None,
    'Path to the fasta file',
)
flags.DEFINE_string(
    'msa',
    None,
    'Path to the msa file',
)

FLAGS = flags.FLAGS


def read_fasta(fasta_path):
    with open(fasta_path, 'r') as f:
        fasta = "".join([a.rstrip() for a in f.readlines() if not a.startswith('>')])
                 
    return fasta    

def protein_to_dict(seq,id='A',modifications=[], unpairedMSA=None,pairedMSA='',templates=[]):

    return {'protein': {
        'id': id,
        'sequence': seq,
        'modifications': modifications,
        'unpairedMsaPath': unpairedMSA,
        'pairedMsaPath': unpairedMSA,
        'templates': templates}}
    
def main(argv):
    #pass
    seq=read_fasta(FLAGS.fasta)
    
    #print(seq)
    #print (seq)
    #print(FLAGS.fasta)
    #print(FLAGS.msa)
    d={}
    d['dialect']='alphafold3'
    d['version']=1
    d['name']=Path(FLAGS.fasta).stem
    d['sequences']=[]


   
    d['modelSeeds']=[random.randrange(2**32 - 1)]
    d['bondedAtomPairs']=None
    d['userCCD']=None
    d['sequences'].append(protein_to_dict(seq,unpairedMSA=FLAGS.msa))
    with open(f'{FLAGS.fasta}.json','w') as f:
        json.dump(d, f, indent=4)




if __name__ == '__main__':
    print('Hello')
    app.run(main)