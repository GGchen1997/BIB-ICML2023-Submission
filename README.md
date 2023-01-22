# BIB
We propose **BIB**: **BI**directional Learning for Offline Model-based **B**iological Sequence Design, which focuses on designing biological sequences to maximize some sequence score.

## Installation

The environment of BIB can be installed by running the following commands:
```bash
pip install torch
pip install transformers
```

## Reproducing Performances on DNA Tasks

We consider two DNA tasks: TFBind8(r) and TFBind10(r). Take the TFBind8(r) task as an example and we can run:
```bash
cd dna
python -u BIB.py --task TFBind8-Exact-v0 --task_mode oracle
```
This command will generate the offline dataset and perform some precomputations.

Then we can run the following command to obtain the experimental results of BIB where gamma_learn=1 activates the *Adaptive*-$\gamma$ module.

```bash
python -u BIB.py --task TFBind8-Exact-v0 --eta_learn 0 --gamma_learn 1  --mode BDI
```

Further ablation studies can verify the effectiveness of forward mapping, backward mapping and the *Adaptive*-$\gamma$ module.
```bash
python -u BIB.py --task TFBind8-Exact-v0 --eta_learn 0 --gamma_learn 0  --mode backward
python -u BIB.py --task TFBind8-Exact-v0 --eta_learn 0 --gamma_learn 0  --mode forward
python -u BIB.py --task TFBind8-Exact-v0 --eta_learn 0 --gamma_learn 0  --mode BDI
```

Last but not least, we can activate the *Adaptive*-$\eta$ module by setting eta_learn=1 to verify its effectiveness.
```bash
python -u BIB.py --task TFBind8-Exact-v0 --eta_learn 1 --gamma_learn 1  --mode BDI
```

The commands for TFBind10(r) are similar.

## Reproducing Performances on Protein Tasks

We consider three protein tasks: avGFP, AAV and E4B. Take the avGFP task as an example and we can run:
```bash
cd protein
python -u BIB.py --task avGFP --task_mode oracle
python -u BIB.py --task avGFP --eta_learn 0 --gamma_learn 1  --mode BDI
python -u BIB.py --task avGFP --eta_learn 0 --gamma_learn 0  --mode backward
python -u BIB.py --task avGFP --eta_learn 0 --gamma_learn 0  --mode forward
python -u BIB.py --task avGFP --eta_learn 0 --gamma_learn 0  --mode BDI
python -u BIB.py --task avGFP --eta_learn 1 --gamma_learn 1  --mode BDI
```

The commands for AAV and E4B are similar.
## Acknowledgements

We thank the pre-trained DNA-BERT model (https://github.com/jerryji1993/DNABERT) for our DNA tasks and the pre-trained Prot-T5 model and Prot-BERT model (https://github.com/agemagician/ProtTrans) for our protein tasks. 
