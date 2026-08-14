"""Converters from common scientific representations to NetworkX graphs.

Chemistry support is optional. Install it with ``pip install 'buhito[chem]'``.
Keeping the RDKit import inside :func:`smiles_to_nx` allows topology-only users
and CI jobs to import Buhito without a chemistry stack.
"""

from __future__ import annotations

import networkx as nx


def smiles_to_nx(
    smiles: str,
    add_hs: bool = False,
    output_2d_pos: bool = False,
):
    """Convert a SMILES string to a labeled undirected NetworkX graph.

    Parameters
    ----------
    smiles:
        Input SMILES string.
    add_hs:
        Add explicit hydrogens before graph construction.
    output_2d_pos:
        Compute and return RDKit 2-D coordinates.

    Returns
    -------
    (graph, positions)
        ``positions`` is ``None`` unless ``output_2d_pos`` is true.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import rdDepictor
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "SMILES conversion requires RDKit. Install it with "
            "`pip install 'buhito[chem]'` or with conda-forge."
        ) from exc

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    if add_hs:
        molecule = Chem.AddHs(molecule)

    positions = None
    if output_2d_pos:
        rdDepictor.Compute2DCoords(molecule)
        conformer = molecule.GetConformer()
        positions = {
            atom.GetIdx(): (
                float(conformer.GetAtomPosition(atom.GetIdx()).x),
                float(conformer.GetAtomPosition(atom.GetIdx()).y),
            )
            for atom in molecule.GetAtoms()
        }

    graph = nx.Graph()
    for atom in molecule.GetAtoms():
        graph.add_node(
            atom.GetIdx(),
            atom_symbol=atom.GetSymbol(),
            atom_key=(atom.GetAtomicNum(), atom.GetFormalCharge()),
        )

    for bond in molecule.GetBonds():
        graph.add_edge(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond_key=str(bond.GetBondType()),
            is_aromatic=bond.GetIsAromatic(),
        )

    return graph, positions
