# Dataset Provenance, Licensing, and Configuration Guide

## 1. Original Reference Repository Dataset (Unavailable)
The reference repository (`https://github.com/sevmardi/Protein-Protein-Networks`) documents five files:
- `HumanPPI.txt` — undirected protein-protein interaction edges (`<Prot_i, Prot_j>`)
- `Functions.txt` — protein functional annotations (`<Prot_i, Func_j>`)
- `Cancers.txt` — known cancer-associated proteins (`<Prot_i>`)
- `Test1.txt` — labeled evaluation proteins
- `Test2.txt` — unlabeled inference proteins

Per the original repository's `readme.md` (*"Dataset is not allowed to be published"*) and `.gitignore` (`data/`), these raw files are **not distributed**. Our project does **not** fabricate these files; instead, we provide an adapter configuration (`config/dataset_original_schema.yaml`) that can ingest them automatically if you possess an authorized local copy, while using verified public biological databases as the primary executable dataset.

---

## 2. Primary Public Biological Dataset (Configured & Verified)

### A. STRING v12.5 (*Homo sapiens* Protein-Protein Interactions & Functional Annotations)
- **Source:** STRING Database v12.5 (`https://string-db.org/cgi/download?species_text=Homo+sapiens`)
- **Files Used:**
  - `9606.protein.links.v12.5.txt.gz` — human PPI edges with `combined_score` (0–1000). Filtered to `combined_score >= 700` (standard STRING high-confidence threshold) and normalized to edge weights in `[0, 1]`.
  - `9606.protein.info.v12.5.txt.gz` — maps Ensembl Protein IDs (`9606.ENSP...`) to canonical HGNC `preferred_name` gene symbols, `protein_size`, and curated biological `annotation` descriptions.
- **License:** Creative Commons Attribution 4.0 International (**CC BY 4.0**).
- **Citation:** Szklarczyk D et al., *The STRING database in 2023: protein–protein association networks and functional enrichment analyses for any sequenced genome of interest*, Nucleic Acids Research (2023).

### B. COSMIC Cancer Gene Census v104 (GRCh38)
- **Source:** Catalogue of Somatic Mutations in Cancer — Wellcome Sanger Institute (`https://cancer.sanger.ac.uk/census`)
- **File Used:**
  - `Cosmic_CancerGeneCensus_v104_GRCh38.tsv.gz` — gold-standard curated catalog of genes with documented somatic/germline mutations causally implicated in cancer (Tier 1 & Tier 2), including `GENE_SYMBOL`, `SYNONYMS`, `ROLE_IN_CANCER` (oncogene / TSG / fusion), `TIER`, and tumour types.
- **License:** COSMIC Academic / Non-Commercial License (free for academic and non-commercial research; registration required at `cancer.sanger.ac.uk`).
- **Citation:** Sondka Z et al., *The COSMIC Cancer Gene Census: describing genetic dysfunction across all human cancers*, Nature Reviews Cancer (2018).

### C. Alternative Open-Access Disease-Gene & Functional Sources
If COSMIC is unavailable in another environment, the schema also supports:
- **DisGeNET** (`https://www.disgenet.org/`) for curated gene-disease associations
- **NCBI Gene / Gene Ontology (GO)** (`https://geneontology.org/`) for standardized GO term annotations
- **UniProtKB/Swiss-Prot** (`https://www.uniprot.org/`) for canonical human protein metadata

---

## 3. Identifier Mapping & Anti-Leakage Controls
1. **Consistent ID Mapping:** Every protein node is keyed by its canonical Ensembl Protein ID (`9606.ENSP...`) and linked to its HGNC `preferred_name` (uppercase normalized). COSMIC genes are matched first by primary `GENE_SYMBOL`, then by unambiguous `SYNONYMS` fallback when no primary match exists.
2. **Functional Text Leak Prevention:** STRING functional descriptions occasionally mention terms such as *"tumor"*, *"cancer"*, *"oncogene"*, or *"carcinoma"*. Before computing TF-IDF/SVD functional features, all cancer/disease-related vocabulary tokens are explicitly scrubbed so the functional feature set reflects genuine molecular biology (kinases, receptors, chromatin remodeling, transcription factors, signaling domains) rather than text label leakage.
