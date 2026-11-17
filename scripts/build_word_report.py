"""Build the advanced report in the updated House Prices sample format.

Only a temporary PDF-render copy substitutes unavailable local fonts. Delivered
DOCX retains Song (宋体) / Times New Roman. No training or baseline edits occur.
"""

from pathlib import Path
import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/advanced"
NAME = "Store_Sales_Data_Analysis_Report"


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def effective_format(paragraph):
    run = paragraph.runs[0]
    style = paragraph.style
    actual = paragraph.paragraph_format
    inherited = style.paragraph_format
    fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
    run_fonts = run._element.get_or_add_rPr().rFonts

    def pick(name, default):
        value = getattr(actual, name)
        if value is None:
            value = getattr(inherited, name)
        return default if value is None else value

    return {
        "latin_font": run.font.name or style.font.name,
        "cjk_font": (run_fonts.get(qn("w:eastAsia")) if run_fonts is not None else None)
        or fonts.get(qn("w:eastAsia")),
        "size_pt": (run.font.size or style.font.size).pt,
        "alignment": int(pick("alignment", WD_ALIGN_PARAGRAPH.JUSTIFY)),
        "line_spacing": float(pick("line_spacing", 1.0)),
        "space_before_pt": pick("space_before", Pt(0)).pt,
        "space_after_pt": pick("space_after", Pt(0)).pt,
        "first_line_indent_pt": pick("first_line_indent", Pt(0)).pt,
        "bold": run.bold if run.bold is not None else bool(style.font.bold),
    }


def compare_reference(report, reference_path):
    reference = Document(reference_path)
    pairs = [
        ("title", reference.paragraphs[0], report.paragraphs[0]),
        (
            "abstract_heading",
            reference.paragraphs[1],
            next(p for p in report.paragraphs if p.text == "摘要"),
        ),
        (
            "body",
            reference.paragraphs[2],
            next(p for p in report.paragraphs if p.text.startswith("本项目以")),
        ),
        (
            "heading_1",
            next(p for p in reference.paragraphs if p.text.startswith("一、")),
            next(p for p in report.paragraphs if p.text.startswith("一、")),
        ),
        (
            "heading_2",
            next(p for p in reference.paragraphs if p.text.startswith("1.1 ")),
            next(p for p in report.paragraphs if p.text.startswith("1.1 ")),
        ),
        (
            "table_header",
            reference.tables[0].cell(0, 0).paragraphs[0],
            report.tables[0].cell(0, 0).paragraphs[0],
        ),
        (
            "table_body",
            reference.tables[0].cell(1, 0).paragraphs[0],
            report.tables[0].cell(1, 0).paragraphs[0],
        ),
    ]
    checks = []
    for name, source, output in pairs:
        expected, actual = effective_format(source), effective_format(output)
        expected_without_bold = {
            key: value for key, value in expected.items() if key != "bold"
        }
        actual_without_bold = {
            key: value for key, value in actual.items() if key != "bold"
        }
        assert expected_without_bold == actual_without_bold, (name, expected, actual)
        assert actual["bold"] is False, name
        checks.append(
            {
                "element": name,
                "reference": expected,
                "report": actual,
                "format_matches_except_requested_no_bold": True,
            }
        )
    return {
        "reference_filename": reference_path.name,
        "reference_sha256": sha(reference_path),
        "requested_override": "All bold removed, including headings, title, keywords and table headers",
        "checks": checks,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument(
        "--reference-docx",
        type=Path,
        help="Compare actual paragraph formatting against the user reference",
    )
    args = parser.parse_args()
    metrics = pd.read_csv(OUT / "metrics.csv")
    leader = pd.read_csv(OUT / "leaderboard.csv").set_index("model").pooled_RMSLE
    sel = json.loads((OUT / "selection.json").read_text())
    reg = metrics.loc[metrics.date.eq("2017-07-31")].set_index("model")
    ens = pd.read_csv(OUT / "seed_ensemble_diagnostics.csv")
    groups = pd.read_csv(OUT / "group_diagnostics.csv")
    zero = groups.loc[
        groups.date.eq("2017-07-31")
        & groups["group"].eq("zero_sales")
        & groups.value.eq("zero")
    ].set_index("model")
    assert sel["tree_model"] == "B3" and sel["weight_tree"] == 1
    assert sel["cpu_config"]["device"] == "cpu"
    assert (
        json.loads((OUT / "independent_verification.json").read_text())[
            "kaggle_public_score"
        ]
        is None
    )
    values = {
        "DEV_BASE": f"{leader.B0:.5f}",
        "DEV_FINAL": f"{leader.B3:.5f}",
        "IMPROVE": f"{1 - leader.B3 / leader.B0:.2%}",
        "REG_BASE": f"{reg.loc['B0', 'RMSLE']:.5f}",
        "REG_FINAL": f"{reg.loc['FINAL', 'RMSLE']:.5f}",
        "SEED_SINGLE": f"{np.sqrt(np.mean(ens.single_seed_RMSLE**2)):.5f}",
        "SEED_MEAN": f"{np.sqrt(np.mean(ens.three_seed_mean_RMSLE**2)):.5f}",
        "MAE": f"{reg.loc['FINAL', 'MAE']:.2f}",
        "RMSE": f"{reg.loc['FINAL', 'RMSE']:.2f}",
        "R2": f"{reg.loc['FINAL', 'R2']:.4f}",
        "REG_GAIN": f"{1 - reg.loc['FINAL', 'RMSLE'] / reg.loc['B0', 'RMSLE']:.2%}",
        "ZERO_BASE": f"{zero.loc['B0', 'RMSLE']:.5f}",
        "ZERO_FINAL": f"{zero.loc['FINAL', 'RMSLE']:.5f}",
    }
    keys = [
        "B0",
        "B1",
        "B3",
        "mlp",
        "mixer2",
        "mixer3",
        "mixer3_long",
        "mixer3_scale",
        "no_pair",
        "no_promo",
        "no_anchor",
    ]
    table_rows = [
        [
            key,
            f"{leader[key]:.5f}",
            "冻结基础模型"
            if key == "B0"
            else "增强树"
            if key.startswith("B")
            else "加长预算"
            if key in ["mixer3_long", "mixer3_scale"]
            else "基础神经预算",
        ]
        for key in keys
    ]
    headers = ["模型", "四窗合并RMSLE", "对照说明"]
    values["MODEL_TABLE"] = "\n".join(
        ["| " + " | ".join(headers) + " |", "| --- | --- | --- |"]
        + ["| " + " | ".join(row) + " |" for row in table_rows]
    )
    source = ROOT / "docs/advanced_word_report.md"
    content = source.read_text()
    for key, value in values.items():
        content = content.replace("{{" + key + "}}", value)
    assert "{{" not in content

    doc = Document()
    section = doc.sections[0]
    # The sample does not specify paper dimensions; A4 with 25.4mm margins.
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.54)
    section.left_margin = section.right_margin = Cm(2.54)
    for name, size in [("Normal", 12), ("Heading 1", 22), ("Heading 2", 16)]:
        style = doc.styles[name]
        style.font.name, style.font.size = "Times New Roman", Pt(size)
        style.font.bold = False
        style.font.cs_bold = False
        style.font.color.rgb = RGBColor(0, 0, 0)
        fonts = style.element.get_or_add_rPr().get_or_add_rFonts()
        fonts.set(qn("w:eastAsia"), "宋体")
        fonts.set(qn("w:cs"), "Times New Roman")
        pf = style.paragraph_format
        pf.first_line_indent = Pt(0)
        pf.space_before = pf.space_after = Pt(0 if name == "Normal" else 5)
        pf.line_spacing = 1.0 if name == "Normal" else 1.5
        pf.alignment = (
            WD_ALIGN_PARAGRAPH.JUSTIFY if name == "Normal" else WD_ALIGN_PARAGRAPH.LEFT
        )
        pf.widow_control = False
        if name != "Normal":
            pf.keep_with_next = pf.keep_together = True
    doc.core_properties.title = "Store Sales：商店销量时序预测（进阶版）"
    doc.core_properties.subject = "新版示例格式，正文为主的进阶实验报告"
    doc.core_properties.author = ""
    lines = content.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line:
            continue
        if line.startswith("|"):
            rows = []
            while line.startswith("|"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                    rows.append(cells)
                if index == len(lines) or not lines[index].strip().startswith("|"):
                    break
                line = lines[index].strip()
                index += 1
            table = doc.add_table(rows=0, cols=len(rows[0]))
            table.style = "Table Grid"
            for number, row in enumerate(rows):
                cells = table.add_row().cells
                for cell, text in zip(cells, row):
                    cell.text = text
                    cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    for run in cell.paragraphs[0].runs:
                        run.bold = False
                table.rows[-1]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
            table.rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        elif line.startswith("### "):
            doc.add_heading(line[4:], level=2)
        elif line.startswith("## "):
            paragraph = doc.add_heading(line[3:], level=1)
            if line[3:] == "摘要":
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif line.startswith("# "):
            paragraph = doc.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.paragraph_format.space_after = Pt(10)
            run = paragraph.add_run(line[2:])
            run.bold, run.font.size = False, Pt(14)
        else:
            paragraph = doc.add_paragraph()
            paragraph.add_run(line).bold = False
            if line.startswith("姓名："):
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    all_paragraphs = list(doc.paragraphs) + [
        paragraph
        for table in doc.tables
        for row in table.rows
        for cell in row.cells
        for paragraph in cell.paragraphs
    ]
    for paragraph in all_paragraphs:
        for run in paragraph.runs:
            run.bold = False
            run.font.cs_bold = False
    docx_path = ROOT / (NAME + ".docx")
    doc.save(docx_path)
    (ROOT / (NAME + ".md")).write_text(content)
    loaded = Document(docx_path)
    assert len(loaded.tables) == 1 and len(loaded.inline_shapes) == 0
    assert len(loaded.tables[0].rows) == len(table_rows) + 1
    for name, size in [("Normal", 12), ("Heading 1", 22), ("Heading 2", 16)]:
        style = loaded.styles[name]
        assert style.font.size.pt == size
        assert style.font.name == "Times New Roman"
        assert style.element.get_or_add_rPr().rFonts.get(qn("w:eastAsia")) == "宋体"
        assert style.font.bold is False and style.font.cs_bold is False
    loaded_paragraphs = list(loaded.paragraphs) + [
        paragraph
        for table in loaded.tables
        for row in table.rows
        for cell in row.cells
        for paragraph in cell.paragraphs
    ]
    assert all(
        run.bold is False and run.font.cs_bold is False
        for paragraph in loaded_paragraphs
        for run in paragraph.runs
    )
    if args.reference_docx:
        comparison = compare_reference(loaded, args.reference_docx)
        (OUT / "word_format_comparison.json").write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2)
        )
    body = "\n".join(p.text for p in loaded.paragraphs)
    assert all(
        value in body
        for value in [values["REG_FINAL"], values["DEV_FINAL"], values["IMPROVE"]]
    )
    pages = None
    if args.pdf:
        import fitz

        office = shutil.which("soffice") or shutil.which("libreoffice")
        assert office, "LibreOffice required for --pdf"
        with tempfile.TemporaryDirectory(prefix="store-sales-sample3-") as tmp:
            tmp = Path(tmp)
            render = Document(docx_path)
            mapping = {
                "宋体": "Noto Serif CJK SC",
                "Times New Roman": "Liberation Serif",
            }
            for root in [render.element, render.styles.element]:
                for fonts in root.iter(qn("w:rFonts")):
                    for attr in ["ascii", "hAnsi", "eastAsia", "cs"]:
                        key = qn("w:" + attr)
                        if fonts.get(key) in mapping:
                            fonts.set(key, mapping[fonts.get(key)])
            render_path = tmp / docx_path.name
            render.save(render_path)
            subprocess.run(
                [
                    office,
                    "-env:UserInstallation=" + (tmp / "profile").as_uri(),
                    "--headless",
                    "--convert-to",
                    "pdf",
                    "--outdir",
                    str(tmp),
                    str(render_path),
                ],
                check=True,
            )
            shutil.copyfile(tmp / (NAME + ".pdf"), ROOT / (NAME + ".pdf"))
        pdf = fitz.open(ROOT / (NAME + ".pdf"))
        pages = len(pdf)
        cjk = lambda text: "".join(re.findall(r"[\u4e00-\u9fff]", text))
        extracted = cjk("".join(page.get_text() for page in pdf))
        for paragraph in loaded.paragraphs:
            expected = cjk(paragraph.text)
            if len(expected) >= 6:
                assert expected in extracted, (
                    "PDF missing Chinese text",
                    paragraph.text,
                )
        assert all(len(page.get_text()) > 100 for page in pdf), "Blank/near-empty page"
    evidence_names = [
        "metrics.csv",
        "leaderboard.csv",
        "selection.json",
        "window_diagnostics.csv",
        "group_diagnostics.csv",
        "seed_ensemble_diagnostics.csv",
        "independent_verification.json",
        "saved_model_verification.json",
    ]
    hashes = {
        NAME + ext: sha(ROOT / (NAME + ext))
        for ext in [".docx", ".md"] + ([".pdf"] if args.pdf else [])
    }
    provenance = {
        "reference": "House_Prices_期末大作业3(1).docx",
        "word_fonts": {"eastAsia": "宋体", "latin": "Times New Roman"},
        "font_sizes_pt": {"title": 14, "body": 12, "heading_1": 22, "heading_2": 16},
        "reference_use": "Typography and prose-first layout; all bold removed per user request",
        "requested_format_overrides": {"bold": False},
        "reference_sha256": sha(args.reference_docx) if args.reference_docx else None,
        "pdf_font_substitution": {
            "宋体": "Noto Serif CJK SC",
            "Times New Roman": "Liberation Serif",
        },
        "pdf_uses_temporary_copy_only": True,
        "report_device": "cpu",
        "kaggle_public_score": None,
        "embedded_figures": [],
        "docx_sha256": hashes[NAME + ".docx"],
        "content_source_sha256": sha(source),
        "evidence_sha256": {
            "outputs/advanced/" + f: sha(OUT / f) for f in evidence_names
        },
    }
    (OUT / "word_report_provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2)
    )
    verification = {
        "pages": pages,
        "tables": 1,
        "embedded_figures": 0,
        "reference_layout_matches_with_requested_no_bold_override": True,
        "reference_document_compared": bool(args.reference_docx),
        "all_bold_removed": True,
        "pdf_cjk_paragraph_text_preserved": args.pdf,
        "no_blank_pages": args.pdf,
        "model_training_performed": False,
        "artifacts_sha256": hashes,
    }
    (OUT / "word_report_verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2)
    )
    print("UPDATED_WORD_REPORT_VERIFIED", verification, flush=True)


if __name__ == "__main__":
    main()
