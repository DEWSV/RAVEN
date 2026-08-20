# RAVEN

**RAVEN (Randomized Atomistic Views with Ensemble Neural Reservoirs)** is a protein–ligand binding-affinity prediction framework that combines frozen random graph representations, an explicit physicochemical interaction fingerprint, and heterogeneous supervised regressors.

<p align="center">
  <img src="RAVEN.png" width="920" alt="RAVEN architecture">
</p>

## Architecture

RAVEN represents each protein–ligand complex as an atomistic heterogeneous graph containing ligand atoms, protein-pocket atoms, intramolecular covalent bonds, and directed cross-interface contacts.

The final model contains four main stages:

1. **Atomistic heterogeneous graph construction**  
   Protein and ligand atoms are encoded at heavy-atom resolution. Cross-interface neighbors within 8 Å provide explicit protein–ligand contact relations.

2. **Frozen multi-view random graph reservoir**  
   Thirty-two independently initialized four-layer heterogeneous MPNN encoders remain fully frozen and generate complementary protein, interaction, and ligand views. An additional frozen global protein–ligand branch provides an asymmetric structural view. The complete raw structural reservoir has **37,632 dimensions**.

3. **Deterministic physicochemical fingerprint**  
   A parallel **788-dimensional** fingerprint summarizes molecular composition, intermolecular geometry, and ten physicochemical interaction channels, including hydrogen bonding, ionic interactions, hydrophobic contacts, aromatic interactions, metal coordination, halogen bonding, and cation–π interactions.

4. **Heterogeneous expert fusion**  
   The structural reservoir and physicochemical fingerprint are decoded by three expert families:
   - **R-MLP ×3**: structure-only neural regressors
   - **J-MLP ×3**: joint structure–physics neural regressors
   - **ExtraTrees ×3**: physics-only tree regressors

   The nine expert predictions are combined using **nonnegative softmax weights fitted on the validation set**. The learned weights are fixed before Test and CASF-2016 evaluation.

## Performance

<p align="center">
  <img src="RAVEN_scatter.png" width="920" alt="RAVEN prediction results">
</p>

| Dataset | N | Pearson r | Spearman ρ | R² | RMSE | MAE |
|---|---:|---:|---:|---:|---:|---:|
| PDBbind 2020R1 Test | 1,749 | 0.7995 | 0.7922 | 0.6341 | 1.2396 | 0.9379 |
| CASF-2016 | 285 | 0.8446 | 0.8407 | 0.6895 | 1.2095 | 0.9243 |

All affinity predictions and error metrics are reported on the dimensionless **pK** scale.

## Key Features

- Fully frozen graph encoders with no affinity-gradient updates
- Multi-seed random structural reservoir instead of end-to-end graph representation training
- Atom-level protein–ligand heterogeneous graphs
- Explicit 788-dimensional physicochemical interaction fingerprint
- Complementary neural and tree-based supervised readers
- Validation-fitted nonnegative softmax ensemble
- Separate reporting on the complete PDBbind Test set and the protected CASF-2016 subset

## Repository

The main model configuration is defined in `config.py`. Model components for graph encoding, physicochemical fingerprint construction, supervised regression, data processing, and inference are provided as separate modules.

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Model paths, dataset paths, runtime options, and inference parameters can be adjusted in the configuration files.

## Citation

If you use RAVEN in academic work, please cite the corresponding manuscript:

> **RAVEN: Frozen Random Graph Reservoirs with Physics-Informed Interaction Fingerprints for Protein–Ligand Binding Affinity Prediction**







中国药科大学
