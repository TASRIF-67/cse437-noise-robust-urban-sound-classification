# IEEE Conference Paper Workspace

This directory contains a generic IEEE conference-style `IEEEtran` project
connected to the verified experimental evidence in `evidence/`.

The paper is a structured draft workspace, not a completed manuscript. Existing
figures and numeric tables are final experiment artifacts; visible drafting
notes and missing-figure boxes identify report content that still requires
writing or analysis.

## Structure

```text
paper/
  main.tex                    IEEEtran entry point
  bibliography/
    references.bib            Verified BibTeX records
  evidence/
    PHASE_3_PROJECT_LOG.md     Complete experiment record
    phase_03/                 Final aggregate CSV/JSON evidence
  figures/
    phase_02/                 Selected dataset/EDA figures
    phase_03/                 Final robustness figures
    FIGURE_MANIFEST.md        Stable figure names and labels
  latex/
    figure_placeholders.tex   Missing-asset fallback
  sections/                   Modular manuscript sections
  tables/                     Verified LaTeX tables
```

## Overleaf

Create an archive with `main.tex` at its root:

```powershell
Compress-Archive -Path paper\* -DestinationPath cse437_ieee_overleaf_project.zip
```

Upload the archive as a new Overleaf project and use pdfLaTeX. The document class
is:

```latex
\documentclass[conference]{IEEEtran}
```

Before submission:

1. replace the author and affiliation placeholders in `main.tex`;
2. replace every visible drafting note with reviewed prose;
3. generate any required figure still marked as a placeholder;
4. add only verified primary-source BibTeX records;
5. confirm every table against `evidence/phase_03/`;
6. check the required page limit and instructor-specific formatting rules.

## Figures

Paper figures use stable semantic names:

```text
fig_p<phase>_<order>_<snake_case_topic>.<extension>
```

Ready assets are stored under `figures/phase_02/` and
`figures/phase_03/`. The `\ProjectFigure` macro renders a labeled box when a
planned file is missing, allowing the draft to compile without creating invalid
placeholder images.

## Evidence policy

The CSV and JSON files under `evidence/phase_03/` are the source of truth for
reported test metrics. Do not replace them with pilot, smoke-run, or
validation-only results. If the final experiment is rerun, refresh the evidence
snapshot and regenerate affected tables deliberately.
