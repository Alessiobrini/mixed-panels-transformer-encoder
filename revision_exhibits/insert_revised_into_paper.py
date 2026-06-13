"""Insert REVISED (lead=2) exhibits immediately BELOW their old counterparts in paper/main.tex.
Tables use the regenerated .tex; figures clone the old block and point at the *_REVISED.pdf files.
Each inserted caption is prefixed with \\textcolor{blue}{REVISED}; labels get a _REV suffix to
avoid duplicate-label clashes. Idempotent: skips an exhibit if its _REV label already exists.
Backs up main.tex to main.tex.prerev first."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "main.tex"

TABLES = {  # label -> generated .tex
    "Tab:evals_simulation": ROOT / "outputs/tables/repl100_lead2_replications.tex",
    "Tab:empirical1": ROOT / "outputs/tables/lead2/empirical1.tex",
    "Tab:empirical2": ROOT / "outputs/tables/lead2/empirical2.tex",
    "Tab:empirical3": ROOT / "outputs/tables/lead2/empirical3.tex",
    "Tab:empirical_abl1": ROOT / "outputs/tables/lead2/empirical_abl1.tex",
    "Tab:empirical_abl2": ROOT / "outputs/tables/lead2/empirical_abl2.tex",
    "Tab:DM_competing": ROOT / "outputs/tables/lead2/dm_competing.tex",
    "Tab:MCS_competing": ROOT / "outputs/tables/lead2/mcs_competing.tex",
    "Tab:DM_ablations": ROOT / "outputs/tables/lead2/dm_ablations.tex",
    "Tab:MCS_ablations": ROOT / "outputs/tables/lead2/mcs_ablations.tex",
}
FIGURES = ["Fig:preds", "Fig:Az_heatmaps_GDP_OUT", "Fig:B_heatmaps_GDP_OUT",
           "Fig:Az_heatmaps_CPI", "Fig:B_heatmaps_AB4"]
BLUE = "\\textcolor{blue}{REVISED} "


def end_of_block(src, label, env):
    li = src.index(f"\\label{{{label}}}")
    ei = src.index(f"\\end{{{env}}}", li) + len(f"\\end{{{env}}}")
    return ei


def begin_of_block(src, label, env):
    li = src.index(f"\\label{{{label}}}")
    return src.rindex(f"\\begin{{{env}}}", 0, li)


def prefix_caption(block: str) -> str:
    return block.replace("\\caption{", "\\caption{" + BLUE, 1)


def rev_label(block: str, label: str) -> str:
    return block.replace(f"\\label{{{label}}}", f"\\label{{{label}_REV}}")


def main():
    src = MAIN.read_text()
    (MAIN.parent / "main.tex.prerev").write_text(src)
    inserted = []

    # process from the BOTTOM up so earlier insertions don't shift later indices
    items = [(lab, "table", TABLES[lab]) for lab in TABLES] + [(lab, "figure", None) for lab in FIGURES]
    # order by position in the file, descending
    items.sort(key=lambda x: src.index(f"\\label{{{x[0]}}}"), reverse=True)

    for label, env, gen in items:
        if f"{label}_REV" in src:
            print(f"skip {label} (already inserted)")
            continue
        ei = end_of_block(src, label, env)
        if env == "table":
            block = gen.read_text().strip()
            block = prefix_caption(block)
            block = rev_label(block, label)  # generated label == original label
        else:
            old = src[begin_of_block(src, label, "figure"):ei]
            block = old.replace(".pdf}", "_REVISED.pdf}")
            block = prefix_caption(block)
            block = rev_label(block, label)
        src = src[:ei] + "\n\n% ---- REVISED (lead=2 rerun) ----\n" + block + "\n" + src[ei:]
        inserted.append(label)
        print(f"inserted REVISED below {label}")

    MAIN.write_text(src)
    print(f"\nInserted {len(inserted)} exhibits. Backup: paper/main.tex.prerev")


if __name__ == "__main__":
    main()
