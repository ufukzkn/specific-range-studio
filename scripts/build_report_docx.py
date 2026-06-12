from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.report_figures import ensure_report_figures

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
EMU_PER_INCH = 914400
MML2OMML_XSL = Path(r"C:\Program Files\Microsoft Office\root\Office16\MML2OMML.XSL")


@dataclass
class ParagraphSpec:
    text: str
    style: str | None = None
    align: Literal["left", "center", "right", "both"] | None = None
    bold: bool = False
    italic: bool = False


@dataclass
class TableSpec:
    rows: list[list[str]]
    col_widths: list[int] | None = None


@dataclass
class ImageSpec:
    path: Path
    caption: str | None = None
    width_inches: float = 6.2


class DocxBuilder:
    def __init__(self) -> None:
        self.body: list[str] = []
        self.media: list[tuple[str, str, Path]] = []
        self.figure_counter = 0
        self.equation_counter = 0
        self._image_index = 0

    def add_paragraph(self, spec: ParagraphSpec) -> None:
        self.body.append(self._paragraph_xml(spec))

    def add_page_break(self) -> None:
        self.body.append("<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>")

    def add_table(self, spec: TableSpec) -> None:
        self.body.append(self._table_xml(spec))

    def add_equation(self, equation_text: str, caption: str | None = None) -> None:
        self.body.append(self._equation_xml(equation_text))
        if caption:
            self.equation_counter += 1
            self.add_paragraph(ParagraphSpec(f"Denklem {self.equation_counter}. {caption}", style="Caption", align="center"))

    def add_image(self, spec: ImageSpec) -> None:
        if not spec.path.exists():
            return
        self._image_index += 1
        rel_id = f"rId{self._image_index + 1}"
        package_name = f"image{self._image_index}{spec.path.suffix.lower()}"
        self.media.append((rel_id, package_name, spec.path))
        self.body.append(self._image_xml(spec.path, rel_id, spec.width_inches))
        if spec.caption:
            self.figure_counter += 1
            self.add_paragraph(ParagraphSpec(f"Şekil {self.figure_counter}. {spec.caption}", style="Caption", align="center"))

    def _paragraph_xml(self, spec: ParagraphSpec) -> str:
        props = []
        if spec.style:
            props.append(f"<w:pStyle w:val=\"{escape(spec.style)}\"/>")
        if spec.align:
            props.append(f"<w:jc w:val=\"{spec.align}\"/>")
        ppr = f"<w:pPr>{''.join(props)}</w:pPr>" if props else ""
        rpr = []
        if spec.bold:
            rpr.append("<w:b/>")
        if spec.italic:
            rpr.append("<w:i/>")
        rpr_xml = f"<w:rPr>{''.join(rpr)}</w:rPr>" if rpr else ""
        runs = []
        for line in spec.text.split("\n"):
            runs.append(f"<w:r>{rpr_xml}<w:t xml:space=\"preserve\">{escape(line)}</w:t></w:r>")
            runs.append("<w:r><w:br/></w:r>")
        if runs:
            runs.pop()
        return f"<w:p>{ppr}{''.join(runs)}</w:p>"

    def _equation_xml(self, equation_text: str) -> str:
        omml = equation_text_to_omml(equation_text)
        if omml is None:
            return self._paragraph_xml(
                ParagraphSpec(equation_text, align="center")
            )
        return (
            "<w:p><w:pPr><w:jc w:val=\"center\"/></w:pPr>"
            f"{omml}"
            "</w:p>"
        )

    def _table_xml(self, spec: TableSpec) -> str:
        widths = spec.col_widths or [2200] * len(spec.rows[0])
        rows_xml = []
        for row_index, row in enumerate(spec.rows):
            cells = []
            for col_index, cell in enumerate(row):
                p = self._paragraph_xml(ParagraphSpec(str(cell), bold=row_index == 0))
                width = widths[min(col_index, len(widths) - 1)]
                cells.append(f"<w:tc><w:tcPr><w:tcW w:w=\"{width}\" w:type=\"dxa\"/></w:tcPr>{p}</w:tc>")
            rows_xml.append("<w:tr>" + "".join(cells) + "</w:tr>")
        return (
            "<w:tbl><w:tblPr><w:tblStyle w:val=\"TableGrid\"/><w:tblW w:w=\"0\" w:type=\"auto\"/>"
            "<w:tblBorders><w:top w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"808080\"/>"
            "<w:left w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"808080\"/>"
            "<w:bottom w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"808080\"/>"
            "<w:right w:val=\"single\" w:sz=\"6\" w:space=\"0\" w:color=\"808080\"/>"
            "<w:insideH w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A0A0A0\"/>"
            "<w:insideV w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"A0A0A0\"/></w:tblBorders></w:tblPr>"
            + "".join(rows_xml) + "</w:tbl>"
        )

    def _image_xml(self, path: Path, rel_id: str, width_inches: float) -> str:
        with Image.open(path) as img:
            pw, ph = img.size
        cx = int(width_inches * EMU_PER_INCH)
        cy = int(cx * (ph / pw))
        doc_pr_id = self._image_index + 100
        name = escape(path.name)
        return f"""
<w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">
<wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{doc_pr_id}" name="{name}"/><a:graphic>
<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr>
<pic:cNvPr id="{doc_pr_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rel_id}"/>
<a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>
""".strip()

    def document_xml(self) -> str:
        return (
            "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            f"<w:document xmlns:w=\"{W_NS}\" xmlns:r=\"{R_NS}\" xmlns:m=\"{M_NS}\" xmlns:wp=\"{WP_NS}\" xmlns:a=\"{A_NS}\" xmlns:pic=\"{PIC_NS}\">"
            f"<w:body>{''.join(self.body)}<w:sectPr/></w:body></w:document>"
        )

    def document_rels_xml(self) -> str:
        rels = ["<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles\" Target=\"styles.xml\"/>"]
        for rel_id, package_name, _ in self.media:
            rels.append(f"<Relationship Id=\"{rel_id}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" Target=\"media/{package_name}\"/>")
        return "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">" + "".join(rels) + "</Relationships>"


def styles_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:after="120" w:line="300" w:lineRule="auto"/></w:pPr><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:after="180"/></w:pPr><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:after="120"/></w:pPr><w:rPr><w:i/><w:sz w:val="24"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="240" w:after="80"/></w:pPr><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:spacing w:before="180" w:after="60"/></w:pPr><w:rPr><w:b/><w:sz w:val="26"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="Caption"/><w:basedOn w:val="Normal"/><w:qFormat/><w:pPr><w:jc w:val="center"/><w:spacing w:after="140"/></w:pPr><w:rPr><w:i/><w:sz w:val="20"/></w:rPr></w:style>
</w:styles>"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>"""


def root_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>"""


def core_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:dcterms="http://purl.org/dc/terms/" xmlns:dcmitype="http://purl.org/dc/dcmitype/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>Havacılık Performans Verilerinde Tabular Transformer Rapor Taslağı</dc:title><dc:creator>OpenAI Codex</dc:creator><cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
</cp:coreProperties>"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"><Application>Microsoft Office Word</Application></Properties>"""


def equation_text_to_omml(equation_text: str) -> str | None:
    mathml = EQUATION_MATHML_MAP.get(equation_text)
    if mathml is None:
        return None
    return mathml_to_omml(mathml)


def mathml_to_omml(mathml: str) -> str:
    if not MML2OMML_XSL.exists():
        raise FileNotFoundError(f"MML2OMML stylesheet not found: {MML2OMML_XSL}")
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as in_file:
        in_file.write(mathml)
        in_path = Path(in_file.name)
    out_path = in_path.with_name(in_path.stem + "_omml.xml")
    script = (
        "$ErrorActionPreference='Stop'; "
        f"$xslt = New-Object System.Xml.Xsl.XslCompiledTransform; "
        f"$xslt.Load('{MML2OMML_XSL.as_posix()}'); "
        f"$xslt.Transform('{in_path.as_posix()}', '{out_path.as_posix()}')"
    )
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            check=True,
            capture_output=True,
            text=True,
        )
        xml_text = out_path.read_text(encoding="utf-8")
    finally:
        if in_path.exists():
            in_path.unlink()
        if out_path.exists():
            out_path.unlink()
    return xml_text.replace("\ufeff", "").replace('<?xml version="1.0" encoding="utf-8"?>', "").strip()


EQUATION_MATHML_MAP: dict[str, str] = {
    r"\hat{y} = f_{\theta}(\mathbf{x})": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mover><mi>y</mi><mo>^</mo></mover><mo>=</mo><msub><mi>f</mi><mi>θ</mi></msub><mo>(</mo><mi>x</mi><mo>)</mo></mrow></math>""",
    r"T_j = b_j + f_j(x_j), \quad T_j \in \mathbb{R}^{d_{model}}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>T</mi><mi>j</mi></msub><mo>=</mo><msub><mi>b</mi><mi>j</mi></msub><mo>+</mo><msub><mi>f</mi><mi>j</mi></msub><mo>(</mo><msub><mi>x</mi><mi>j</mi></msub><mo>)</mo><mo>,</mo><msub><mi>T</mi><mi>j</mi></msub><mo>∈</mo><msup><mi>ℝ</mi><mrow><msub><mi>d</mi><mrow><mi>m</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi></mrow></msub></mrow></msup></mrow></math>""",
    r"T_j^{(num)} = b_j^{(num)} + x_j^{(num)} W_j^{(num)}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>T</mi><mi>j</mi><mrow><mi>n</mi><mi>u</mi><mi>m</mi></mrow></msubsup><mo>=</mo><msubsup><mi>b</mi><mi>j</mi><mrow><mi>n</mi><mi>u</mi><mi>m</mi></mrow></msubsup><mo>+</mo><msubsup><mi>x</mi><mi>j</mi><mrow><mi>n</mi><mi>u</mi><mi>m</mi></mrow></msubsup><msubsup><mi>W</mi><mi>j</mi><mrow><mi>n</mi><mi>u</mi><mi>m</mi></mrow></msubsup></mrow></math>""",
    r"T_j^{(cat)} = b_j^{(cat)} + e_j^{T} W_j^{(cat)}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msubsup><mi>T</mi><mi>j</mi><mrow><mi>c</mi><mi>a</mi><mi>t</mi></mrow></msubsup><mo>=</mo><msubsup><mi>b</mi><mi>j</mi><mrow><mi>c</mi><mi>a</mi><mi>t</mi></mrow></msubsup><mo>+</mo><msubsup><mi>e</mi><mi>j</mi><mi>T</mi></msubsup><msubsup><mi>W</mi><mi>j</mi><mrow><mi>c</mi><mi>a</mi><mi>t</mi></mrow></msubsup></mrow></math>""",
    r"T = stack([T_1,\ldots,T_k]) \in \mathbb{R}^{k \times d_{model}}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>T</mi><mo>=</mo><mi>stack</mi><mo>(</mo><mo>[</mo><msub><mi>T</mi><mn>1</mn></msub><mo>,</mo><mo>…</mo><mo>,</mo><msub><mi>T</mi><mi>k</mi></msub><mo>]</mo><mo>)</mo><mo>∈</mo><msup><mi>ℝ</mi><mrow><mi>k</mi><mo>×</mo><msub><mi>d</mi><mrow><mi>m</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi></mrow></msub></mrow></msup></mrow></math>""",
    r"Attention(Q,K,V)=softmax\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>Attention</mi><mo>(</mo><mi>Q</mi><mo>,</mo><mi>K</mi><mo>,</mo><mi>V</mi><mo>)</mo><mo>=</mo><mi>softmax</mi><mo>(</mo><mfrac><mrow><mi>Q</mi><msup><mi>K</mi><mi>T</mi></msup></mrow><mrow><msqrt><msub><mi>d</mi><mi>k</mi></msub></msqrt></mrow></mfrac><mo>)</mo><mi>V</mi></mrow></math>""",
    r"FFN(z)=W_2 \sigma(W_1 z + b_1)+b_2": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>FFN</mi><mo>(</mo><mi>z</mi><mo>)</mo><mo>=</mo><msub><mi>W</mi><mn>2</mn></msub><mi>σ</mi><mo>(</mo><msub><mi>W</mi><mn>1</mn></msub><mi>z</mi><mo>+</mo><msub><mi>b</mi><mn>1</mn></msub><mo>)</mo><mo>+</mo><msub><mi>b</mi><mn>2</mn></msub></mrow></math>""",
    r"Z' = Z + Dropout(MHSA(LN(Z)))": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>Z</mi><mo>′</mo></msup><mo>=</mo><mi>Z</mi><mo>+</mo><mi>Dropout</mi><mo>(</mo><mi>MHSA</mi><mo>(</mo><mi>LN</mi><mo>(</mo><mi>Z</mi><mo>)</mo><mo>)</mo><mo>)</mo></mrow></math>""",
    r"Z_{new} = Z' + Dropout(FFN(LN(Z')))": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>Z</mi><mrow><mi>n</mi><mi>e</mi><mi>w</mi></mrow></msub><mo>=</mo><msup><mi>Z</mi><mo>′</mo></msup><mo>+</mo><mi>Dropout</mi><mo>(</mo><mi>FFN</mi><mo>(</mo><mi>LN</mi><mo>(</mo><msup><mi>Z</mi><mo>′</mo></msup><mo>)</mo><mo>)</mo><mo>)</mo></mrow></math>""",
    r"P_{block} \approx 4d_{model}^{2} + 2d_{model}d_{ff}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>P</mi><mrow><mi>b</mi><mi>l</mi><mi>o</mi><mi>c</mi><mi>k</mi></mrow></msub><mo>≈</mo><mn>4</mn><msup><msub><mi>d</mi><mrow><mi>m</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi></mrow></msub><mn>2</mn></msup><mo>+</mo><mn>2</mn><msub><mi>d</mi><mrow><mi>m</mi><mi>o</mi><mi>d</mi><mi>e</mi><mi>l</mi></mrow></msub><msub><mi>d</mi><mrow><mi>f</mi><mi>f</mi></mrow></msub></mrow></math>""",
    r"P_{total} \approx L \cdot P_{block} + P_{tokenizer} + P_{head}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msub><mi>P</mi><mrow><mi>t</mi><mi>o</mi><mi>t</mi><mi>a</mi><mi>l</mi></mrow></msub><mo>≈</mo><mi>L</mi><mo>⋅</mo><msub><mi>P</mi><mrow><mi>b</mi><mi>l</mi><mi>o</mi><mi>c</mi><mi>k</mi></mrow></msub><mo>+</mo><msub><mi>P</mi><mrow><mi>t</mi><mi>o</mi><mi>k</mi><mi>e</mi><mi>n</mi><mi>i</mi><mi>z</mi><mi>e</mi><mi>r</mi></mrow></msub><mo>+</mo><msub><mi>P</mi><mrow><mi>h</mi><mi>e</mi><mi>a</mi><mi>d</mi></mrow></msub></mrow></math>""",
    r"J(\theta)=w_1 \frac{RMSE(\theta)}{RMSE_{ref}}+w_2 \frac{T_{inf}(\theta)}{T_{ref}}+w_3 \frac{S(\theta)}{S_{ref}}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>J</mi><mo>(</mo><mi>θ</mi><mo>)</mo><mo>=</mo><msub><mi>w</mi><mn>1</mn></msub><mfrac><mrow><mi>RMSE</mi><mo>(</mo><mi>θ</mi><mo>)</mo></mrow><mrow><msub><mi>RMSE</mi><mrow><mi>r</mi><mi>e</mi><mi>f</mi></mrow></msub></mrow></mfrac><mo>+</mo><msub><mi>w</mi><mn>2</mn></msub><mfrac><mrow><msub><mi>T</mi><mrow><mi>i</mi><mi>n</mi><mi>f</mi></mrow></msub><mo>(</mo><mi>θ</mi><mo>)</mo></mrow><mrow><msub><mi>T</mi><mrow><mi>r</mi><mi>e</mi><mi>f</mi></mrow></msub></mrow></mfrac><mo>+</mo><msub><mi>w</mi><mn>3</mn></msub><mfrac><mrow><mi>S</mi><mo>(</mo><mi>θ</mi><mo>)</mo></mrow><mrow><msub><mi>S</mi><mrow><mi>r</mi><mi>e</mi><mi>f</mi></mrow></msub></mrow></mfrac></mrow></math>""",
    r"RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>RMSE</mi><mo>=</mo><msqrt><mrow><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><msup><mrow><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>)</mo></mrow><mn>2</mn></msup></mrow></msqrt></mrow></math>""",
    r"MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>MAE</mi><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mo>|</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>|</mo></mrow></math>""",
    r"R^{2}=1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><msup><mi>R</mi><mn>2</mn></msup><mo>=</mo><mn>1</mn><mo>-</mo><mfrac><mrow><munder><mo>∑</mo><mi>i</mi></munder><msup><mrow><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub><mo>)</mo></mrow><mn>2</mn></msup></mrow><mrow><munder><mo>∑</mo><mi>i</mi></munder><msup><mrow><mo>(</mo><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><mover><mi>y</mi><mo>¯</mo></mover><mo>)</mo></mrow><mn>2</mn></msup></mrow></mfrac></mrow></math>""",
    r"MAPE=\frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|": """<math xmlns="http://www.w3.org/1998/Math/MathML"><mrow><mi>MAPE</mi><mo>=</mo><mfrac><mn>1</mn><mi>n</mi></mfrac><munderover><mo>∑</mo><mrow><mi>i</mi><mo>=</mo><mn>1</mn></mrow><mi>n</mi></munderover><mo>|</mo><mfrac><mrow><msub><mi>y</mi><mi>i</mi></msub><mo>-</mo><msub><mover><mi>y</mi><mo>^</mo></mover><mi>i</mi></msub></mrow><mrow><msub><mi>y</mi><mi>i</mi></msub></mrow></mfrac><mo>|</mo></mrow></math>""",
}


def write_docx(output_path: Path, builder: DocxBuilder) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml())
        archive.writestr("_rels/.rels", root_rels_xml())
        archive.writestr("word/document.xml", builder.document_xml())
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/_rels/document.xml.rels", builder.document_rels_xml())
        archive.writestr("docProps/core.xml", core_xml())
        archive.writestr("docProps/app.xml", app_xml())
        for _, package_name, path in builder.media:
            archive.write(path, f"word/media/{package_name}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_overall_summary(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return next(csv.DictReader(handle), {})


def load_clean_dataset(root: Path) -> pd.DataFrame:
    return pd.read_csv(root / "data" / "processed" / "combined_specific_range.csv")


TRACKED_SECTION_HEADINGS = [
    "1. Giriş ve Motivasyon",
    "2. Veri ve Problemin Tanımı",
    "3. Mimari Bileşenler",
    "4. Katmanlar ve Veri Akışı",
    "5. Tokenizasyon Matematiği ve Sayısal/Kategorik Örnekler",
    "6. Transformer Encoder Bloğu: MHSA, FFN, LayerNorm ve Residual Akışı",
]


def normalize_spaced_ocr_line(line: str) -> str:
    line = line.rstrip()
    if not line:
        return ""
    stripped = line.strip()
    tokens = stripped.split()
    single_char_ratio = (sum(len(token) == 1 for token in tokens) / len(tokens)) if tokens else 0.0
    if len(tokens) >= 6 and single_char_ratio >= 0.55:
        protected = re.sub(r"\s{2,}", " | ", stripped)
        compact = protected.replace(" ", "").replace("|", " ")
        return re.sub(r"\s+", " ", compact).strip()
    return re.sub(r"\s+", " ", stripped)


def extract_ocr_sections(root: Path) -> dict[str, str]:
    extract_path = root / "pdf_extract_utf8.txt"
    if not extract_path.exists():
        return {}

    skip_patterns = [
        "merhaba@harikasite.web.tr",
        "HAVACILIKTA YAPAY ZEKA ARAŞTIRMALARI TOPLULUĞU",
        "PROJE :",
        "GÖREV :",
        "TARİH :",
        "YÖNTEM",
        "--- PAGE",
        "+ 0",
    ]
    heading_map = {
        "Giriş ve Motivasyon": "1. Giriş ve Motivasyon",
        "Veri ve problem tanımı": "2. Veri ve Problemin Tanımı",
        "Mimari bileşenler": "3. Mimari Bileşenler",
        "Katmanlar ve veri akışı": "4. Katmanlar ve Veri Akışı",
        "Tokenizasyon matematiği ve sayısal/kategorik örnekler": "5. Tokenizasyon Matematiği ve Sayısal/Kategorik Örnekler",
        "MHSA, FFN, LayerNorm ve residual akışı": "6. Transformer Encoder Bloğu: MHSA, FFN, LayerNorm ve Residual Akışı",
    }
    sections = {value: [] for value in heading_map.values()}
    current_heading: str | None = None

    for raw_line in extract_path.read_text(encoding="utf-8").splitlines():
        if any(pattern in raw_line for pattern in skip_patterns):
            continue
        normalized = normalize_spaced_ocr_line(raw_line)
        if not normalized:
            continue
        matched_heading = next((value for key, value in heading_map.items() if normalized.lower() == key.lower()), None)
        if matched_heading:
            current_heading = matched_heading
            continue
        if current_heading is not None:
            sections[current_heading].append(normalized)

    return {heading: " ".join(lines).strip() for heading, lines in sections.items() if lines}


def extract_report_sections_from_builder(builder: DocxBuilder) -> dict[str, str]:
    xml_text = builder.document_xml()
    root = ET.fromstring(xml_text)
    namespace = {"w": W_NS}
    sections: dict[str, list[str]] = {heading: [] for heading in TRACKED_SECTION_HEADINGS}
    current_heading: str | None = None

    for paragraph in root.findall(".//w:body/w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        if not text:
            continue
        if text in TRACKED_SECTION_HEADINGS:
            current_heading = text
            continue
        if current_heading is not None:
            sections[current_heading].append(text)

    return {heading: " ".join(lines).strip() for heading, lines in sections.items() if lines}


def write_gap_matrix(root: Path, builder: DocxBuilder) -> Path:
    ocr_sections = extract_ocr_sections(root)
    report_sections = extract_report_sections_from_builder(builder)
    rows: list[dict[str, object]] = []
    for heading in TRACKED_SECTION_HEADINGS:
        ocr_text = ocr_sections.get(heading, "")
        report_text = report_sections.get(heading, "")
        ocr_words = len(ocr_text.split())
        report_words = len(report_text.split())
        ratio = round(report_words / ocr_words, 3) if ocr_words else None
        rows.append(
            {
                "section": heading,
                "ocr_word_count": ocr_words,
                "report_word_count": report_words,
                "coverage_ratio": ratio,
                "ocr_excerpt": ocr_text[:320],
                "report_excerpt": report_text[:320],
            }
        )

    output_dir = root / "artifacts" / "report_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "report_gap_matrix.json"
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def find_first_existing(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_global_summary_table(root: Path) -> list[list[str]]:
    df = load_clean_dataset(root)
    rows = [["Veri seti", "Satır sayısı", "Altitude range", "Altitude levels", "Fuel Flow min-max", "Fuel Flow p01-p99", "Specific Range min-max", "Specific Range p01-p99"]]
    for engine, group in df.groupby("engine_type"):
        rows.append([
            engine,
            str(len(group)),
            f"{int(group['altitude'].min())}-{int(group['altitude'].max())} ft",
            str(int(group['altitude'].nunique())),
            f"{group['fuel_flow'].min():.2f}-{group['fuel_flow'].max():.2f}",
            f"{group['fuel_flow'].quantile(0.01):.2f}-{group['fuel_flow'].quantile(0.99):.2f}",
            f"{group['specific_range'].min():.6f}-{group['specific_range'].max():.6f}",
            f"{group['specific_range'].quantile(0.01):.6f}-{group['specific_range'].quantile(0.99):.6f}",
        ])
    return rows


def build_altitude_distribution_table(root: Path) -> list[list[str]]:
    df = load_clean_dataset(root)
    counts = df.groupby(["altitude", "engine_type"]).size().unstack(fill_value=0).sort_index()
    rows = [["Altitude (ft)", "One Engine (n)", "Two Engine (n)", "Toplam (n)"]]
    for altitude, row in counts.iterrows():
        one_engine = int(row.get("one_engine", 0))
        two_engine = int(row.get("two_engine", 0))
        rows.append([f"{int(altitude)}", str(one_engine), str(two_engine), str(one_engine + two_engine)])
    rows.append(["Toplam", str(int(counts.get("one_engine", pd.Series(dtype=int)).sum())), str(int(counts.get("two_engine", pd.Series(dtype=int)).sum())), str(int(counts.sum().sum()))])
    return rows


def build_metrics_table(root: Path) -> list[list[str]]:
    xgb = load_json(root / "artifacts" / "xgboost" / "metrics.json")
    ft = load_json(root / "artifacts" / "ft_transformer" / "metrics.json")
    rows = [["Model", "Val RMSE", "Test RMSE", "Test MAE", "Test R²", "Test MAPE"]]
    for model_name, payload in [("XGBoost", xgb), ("FT-Transformer", ft)]:
        metrics = payload.get("metrics", {})
        val = metrics.get("validation", {})
        test = metrics.get("test", {})
        if test:
            rows.append([model_name, f"{val.get('rmse', 0):.6f}", f"{test.get('rmse', 0):.6f}", f"{test.get('mae', 0):.6f}", f"{test.get('r2', 0):.6f}", f"{test.get('mape', 0):.4f}"])
    return rows


def build_row_level_table(root: Path) -> list[list[str]]:
    xgb = load_overall_summary(root / "artifacts" / "xgboost" / "reports" / "xgboost_overall_summary.csv")
    ft = load_overall_summary(root / "artifacts" / "ft_transformer" / "reports" / "ft_transformer_overall_summary.csv")
    rows = [["Model", "Rows", "RMSE", "MAE", "R²", "MAPE", "Max Abs Error"]]
    for model_name, payload in [("XGBoost", xgb), ("FT-Transformer", ft)]:
        if payload:
            rows.append([model_name, payload.get("rows", "-"), f"{float(payload.get('rmse', 0.0)):.6f}", f"{float(payload.get('mae', 0.0)):.6f}", f"{float(payload.get('r2', 0.0)):.6f}", f"{float(payload.get('mape', 0.0)):.4f}", f"{float(payload.get('max_abs_error', 0.0)):.6f}"])
    return rows


def build_report(root: Path) -> DocxBuilder:
    builder = DocxBuilder()
    report_figures = ensure_report_figures(root)
    builder.add_paragraph(ParagraphSpec("HAVACILIK PERFORMANS VERİLERİNDE TABULAR TRANSFORMER RAPOR TASLAĞI", style="Title"))
    builder.add_paragraph(ParagraphSpec("FT-Transformer, XGBoost ve PSO tabanlı çoklu amaçlı modelleme", style="Subtitle"))
    builder.add_paragraph(
        ParagraphSpec(
            "Bu sürüm, sohbet içinde bölüm bölüm iyileştirilen rapor metinlerinin daha akademik düzenlenmiş halidir. "
            "Formüller Word içinde equation olarak yerleştirilmiş, ayrıca üretilen rapor çıktılarından seçili görseller de belgeye eklenmiştir.",
            align="center",
            italic=True,
        )
    )
    builder.add_page_break()

    builder.add_paragraph(ParagraphSpec("1. Giriş ve Motivasyon", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Uçak aviyoniklerinde specific range gibi performans büyüklükleri çoğu zaman tablo halinde saklanır. Ancak bu tablolar genellikle "
            "yalnızca belirli irtifa ve uçuş koşullarını içerir. Uçak ara bir değerde, örneğin 11,000 ft gibi tabloda doğrudan bulunmayan "
            "bir noktada uçtuğunda, sistemin bu değeri dolaylı yöntemlerle tahmin etmesi gerekir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Havacılık performans kestirimi güçlü biçimde doğrusal olmayan bir problem sınıfıdır. Mach rejimi, brüt ağırlık, sürükleme durumu "
            "ve yakıt akışı gibi değişkenler yalnızca tek tek etkili olmaz; aynı zamanda birbirlerini rejime bağlı olarak değiştirir. Bu nedenle "
            "sorun, tekil tablo okuma probleminden çok, bölgesel davranışları olan çok boyutlu bir performans yüzeyinin temsil edilmesi problemidir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Klasik yaklaşım, lookup table ve nümerik interpolasyon kullanmaktır. Bu yöntem pratik ve açıklanabilir olsa da, çok boyutlu ve "
            "doğrusal olmayan uçuş ilişkilerini her zaman yeterince iyi temsil edemez. Ayrıca doğruluk arttırılmak istendiğinde tablo boyutu, "
            "bellek ihtiyacı ve bakım maliyeti de hızla büyür. Bu çalışmanın çıkış noktası, bu ara değer tahminini öğrenilebilir bir modelle "
            "yapmanın ne kadar mümkün olduğunu göstermektir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Tarihsel olarak lookup table yaklaşımı güvenilir kabul edilmiştir; çünkü deterministik davranır, açıkça denetlenebilir ve gömülü sistemlerde "
            "uygulanması görece kolaydır. Ancak özellikle transonik bölge gibi eğriliğin yüksek olduğu rejimlerde doğrusal ya da çoklu doğrusal "
            "interpolasyon hatayı büyütebilir. Tablo çözünürlüğünü artırmak bu hatayı azaltabilse de, bu kez bellek ve erişim maliyeti belirgin biçimde artar."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu nedenle temel mühendislik sorusu yalnızca 'daha doğru tahmin nasıl yapılır?' değildir. Asıl soru, aynı performans bilgisinin daha kompakt, "
            "daha genellenebilir ve donanım-kısıtlı ortamlarda daha yönetilebilir bir modelle temsil edilip edilemeyeceğidir. Raporun geri kalanında "
            "FT-Transformer bu soruya verilen aday cevaplardan biri olarak ele alınmaktadır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu nedenle proje, iki hedefi birlikte ele almaktadır: birincisi, specific range tahminini modern tabular öğrenme yöntemleriyle "
            "gerçekleştirmek; ikincisi ise bu modellerin donanım maliyeti açısından ne kadar uygulanabilir olduğunu incelemektir. Burada FT-Transformer "
            "ana model, XGBoost ise güçlü bir kıyas çizgisi olarak kullanılmıştır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Buradaki amaç Transformer mimarisini peşinen üstün ilan etmek değildir. Amaç, FT-Transformer’ın havacılık performans verileri için neden anlamlı "
            "bir aday model ailesi olduğunu göstermek, bunu XGBoost gibi güçlü bir referans yöntemle adil biçimde karşılaştırmak ve aynı zamanda modeli "
            "PSO ile donanım-farkındalıklı biçimde optimize etmektir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu vurgu önemlidir; çünkü tabular derin öğrenme literatüründe Transformer tabanlı modellerin her veri probleminde ağaç tabanlı yöntemlere "
            "evrensel üstünlük sağladığına dair kesin bir uzlaşı bulunmamaktadır. Bu yüzden FT-Transformer burada bir 'iddia' olarak değil, havacılık "
            "performans verileri için gerekçelendirilen ve güçlü baseline'larla karşılaştırılan bir aday yaklaşım olarak ele alınmaktadır."
        )
    )
    builder.add_image(
        ImageSpec(
            report_figures["motivation"],
            "Tablo tabanlı performans verilerinden ara deger specific range tahminine gecis problemi. Sema proje icin yeniden cizilmistir.",
            6.4,
        )
    )

    builder.add_paragraph(ParagraphSpec("2. Veri ve Problemin Tanımı", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Bu çalışma, One Engine ve Two Engine veri kaynaklarından elde edilen specific range tablolarını çoklu irtifa rejimlerinde "
            "öğrenilebilir bir regresyon problemine dönüştürmektedir. Her satır, belirli bir uçuş koşulunu temsil eder; hedef değişken "
            "specific_range, girdiler ise altitude, gross_weight, drag_index, mach, fuel_flow ve engine_type olarak tanımlanmıştır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Veri kaynağı olarak kullanılan Excel dosyaları, NATOPS performans grafiklerinden sayısallaştırılmış arama tabloları gibi düşünülebilir. "
            "Kullanıcı senaryosunda bu grafiklere ağırlık, drag index ve uçuş koşulları ile girilir; specific range ve fuel flow değerleri okunur, "
            "gerekirse eğriler arasında interpolasyon yapılarak ara çalışma noktaları elde edilir. Projede kullanılan veri setleri bu tabular performans "
            "bilgisinin öğrenilebilir biçime dönüştürülmüş karşılığıdır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Başka bir ifadeyle model, tek bir tablo satırını ezberlemek yerine, bu değişkenlerin birlikte oluşturduğu çok boyutlu performans "
            "yüzeyini öğrenmeye çalışır. Böylece hem tablo içinde bulunan noktalar hem de iki tablo noktası arasındaki ara değerler için tahmin "
            "üretilebilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Çalışmada kullanılan iki veri kaynağı farklı operasyon modlarını temsil etmektedir: `One_Engine_Data.xlsx` tek motor, "
            "`Two_Engine_Data.xlsx` ise çift motor çalışma uzayına karşılık gelir. Her iki veri setinin birlikte ele alınabilmesi ve modelin modlar "
            "arası davranış farklarını öğrenebilmesi için birleşik veri yapısına `engine_type` etiketi eklenmiştir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Veri setleri birden fazla irtifa seviyesini içermektedir. Temizlenmiş veri içinde one-engine tarafı 35,000 ft seviyesine kadar, "
            "two-engine tarafı ise 50,000 ft seviyesine kadar uzanmaktadır. Dolayısıyla üst irtifa bölgelerinde örnek yoğunluğu motor konfigürasyonuna "
            "göre farklılaşmaktadır; bu durum hem genelleme davranışını hem de veri kapsamını yorumlarken dikkate alınmalıdır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Veri temizleme ve standardizasyon aşamasında farklı sheet ve workbook yapılarından gelen kayıtlar ortak bir şemaya dönüştürülmüştür. "
            "Ondalık ayırıcı farklılıkları, kolon isimlerindeki küçük varyasyonlar ve irtifa bilgisinin sheet bağlamından okunması gibi adımlar "
            "deterministik biçimde tanımlanmış; böylece eğitim ve çıkarım aşamalarında aynı veri temsili korunmuştur."
        )
    )
    builder.add_paragraph(ParagraphSpec("Regresyon problemi genel biçimde aşağıdaki şekilde ifade edilebilir:", style="Heading2"))
    builder.add_equation(r"\hat{y} = f_{\theta}(\mathbf{x})", "Regresyon hedef fonksiyonu.")
    builder.add_paragraph(
        ParagraphSpec(
            "Burada x vektörü uçuş ve konfigürasyon girdilerini, y_hat ise tahmin edilen specific range değerini temsil etmektedir."
        )
    )
    builder.add_paragraph(ParagraphSpec("Genel veri özeti", style="Heading2"))
    builder.add_paragraph(
        ParagraphSpec(
            "Veri setlerinin yalnızca toplam büyüklüğünü değil, aynı zamanda irtifa kapsamını ve hedef değişken aralıklarını görünür kılmak "
            "amacıyla temizlenmiş veri seti üzerinden global bir özet tablo hazırlanmıştır."
        )
    )
    builder.add_table(TableSpec(build_global_summary_table(root), col_widths=[1600, 1300, 1500, 1300, 1900, 1900, 1900, 1900]))
    builder.add_paragraph(ParagraphSpec("İrtifa seviyelerine göre örnek dağılımı", style="Heading2"))
    builder.add_paragraph(
        ParagraphSpec(
            "Aşağıdaki tablo, temizlenmiş birleşik veri setinde irtifa seviyelerine göre örnek dağılımını göstermektedir. Bu yapı, veri kapsamının "
            "irtifa boyunca homojen olmadığını ve model davranışının farklı rejimlerde ayrıca değerlendirilmesi gerektiğini ortaya koymaktadır."
        )
    )
    builder.add_table(TableSpec(build_altitude_distribution_table(root), col_widths=[1800, 1800, 1800, 1800]))
    builder.add_image(
        ImageSpec(
            report_figures["altitude_distribution"],
            "Temizlenmis veri setinde irtifa seviyelerine gore ornek dagilimi. Grafik, veri kapsamindaki rejim farklarini gostermek icin proje icin yeniden uretilmistir.",
            6.4,
        )
    )

    builder.add_paragraph(ParagraphSpec("3. Mimari Bileşenler", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Tabular Transformer ailesi içinde iki ana yaklaşım öne çıkar: TabTransformer ve FT-Transformer. TabTransformer, özellikle kategorik "
            "özellikleri bağlamsal embeddinglere dönüştürmeye odaklanır. FT-Transformer ise bir adım daha ileri giderek sayısal ve kategorik tüm "
            "özellikleri ortak tokenlar halinde temsil eder. Bu fark önemlidir; çünkü bizim problemimizde irtifa, Mach, yakıt akışı ve ağırlık gibi "
            "sayısal değişkenler doğrudan modelin merkezindedir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu projede veri yapısı büyük ölçüde sayısal özelliklerden oluştuğu ve problem çoklu irtifa rejimlerine yayılan bir performans yüzeyi "
            "olduğu için FT-Transformer daha uygun adaydır. Çünkü FT-Transformer, yalnızca tek tek kolonların etkisini değil; irtifa–Mach, "
            "gross_weight–drag_index veya fuel_flow–engine_type gibi etkileşimleri de self-attention mekanizması içinde birlikte modelleyebilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu ayrım sezgisel olarak şöyle düşünülebilir: TabTransformer, kategorik kolonları daha akıllı embeddinglerle temsil etmeye ağırlık verir. "
            "FT-Transformer ise sayısal kolonları da aynı ciddiyetle ele alır ve hepsini ortak token dizisi içine taşır. Havacılık performans verilerinde "
            "esas yük sayısal değişkenlerde olduğu için bu ikinci yaklaşım daha doğrudan bir uyum göstermektedir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Kısacası bu bölümün özeti şudur: TabTransformer daha çok kategorik kolonları bağlamsallaştırırken, FT-Transformer tüm kolonları ortak "
            "bir dilde konuşturur. Bu da bizim veri tipimiz için daha doğal bir eşleşme sunar."
        )
    )

    builder.add_paragraph(ParagraphSpec("4. Katmanlar ve Veri Akışı", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Modelin veri akışı sezgisel olarak dört adımda düşünülebilir: önce veri temizlenir ve ortak biçime getirilir; sonra her kolon bir token "
            "veya embedding temsiline dönüştürülür; ardından bu tokenlar Transformer encoder bloklarından geçirilir; son olarak [CLS] tokeninden "
            "üretilen temsil regresyon başlığına verilerek specific range tahmini elde edilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Burada `[CLS]` tokeni, tüm örneğin özet temsili gibi davranır. Encoder katmanları boyunca diğer kolonlarla birlikte güncellenir ve sonunda "
            "tek bir sayısal tahmin üretecek regresyon başlığına bağlanır. Bu nedenle `[CLS]`, tek tek kolonların ötesinde satırın bütün davranışını "
            "taşıyan kompakt bir temsil olarak düşünülebilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Buradaki kritik fikir şudur: klasik bir tabloda her kolon ayrı dururken, FT-Transformer bu kolonları aynı temsil uzayına taşır. Böylece model, "
            "örneğin bir satırdaki irtifa değişince Mach ile fuel_flow ilişkisinin de nasıl değiştiğini birlikte görebilir. Uçuş performansı gibi kuvvetli "
            "değişken etkileşimleri içeren problemlerde bu, önemli bir avantajdır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Daha ayrıntılı bakıldığında her örnek önce sayısal ve kategorik kolonlar halinde ayrıştırılır. Ardından her kolon için bir token üretilir, "
            "bu tokenlar istiflenerek ortak bir giriş matrisi oluşturulur ve diziye bir `[CLS]` token eklenir. Sonraki aşamada bu dizi ardışık encoder "
            "bloklarından geçirilir; her blokta MHSA ve FFN katmanları residual ve normalization adımlarıyla birlikte çalışır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu akışın asıl avantajı, tekil kolonların ayrı ayrı işlenmesi yerine satırın tamamının birlikte değerlendirilmesidir. Böylece model, örneğin "
            "aynı Mach değerinin farklı irtifalarda neden farklı specific range davranışı verdiğini ya da aynı drag index’in ağırlık değişimiyle neden "
            "başka bir rejime kaydığını tek bir mimari akış içinde öğrenebilir."
        )
    )
    builder.add_image(
        ImageSpec(
            report_figures["ft_pipeline"],
            "FT-Transformer genel veri akis semasi. Sema, Gorishniy vd. (2021) Fig. 1-2 mantigi temel alinarak proje icin yeniden cizilmistir.",
            6.5,
        )
    )

    builder.add_paragraph(ParagraphSpec("5. Tokenizasyon Matematiği ve Sayısal/Kategorik Örnekler", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "FT-Transformer’da her kolon, modelin anlayabileceği sabit uzunlukta bir vektöre çevrilir. Bu işleme tokenizasyon denir. Sayısal "
            "özelliklerde bu dönüşüm, ham sayının öğrenilebilir bir projeksiyonla çarpılması şeklinde düşünülebilir. Kategorik özelliklerde ise "
            "ilgili kategori için uygun embedding satırı seçilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu projede engine_type kategorik, altitude, gross_weight, mach ve fuel_flow ise sayısal özellikler olarak ele alınmıştır. Drag index "
            "değişkeni veri temsil tercihlerine göre sayısal ya da kategorik şekilde modellenebilir; pratikte iki yaklaşım da düşünülebilir. "
            "Ama hangi temsil seçilirse seçilsin amaç aynıdır: farklı kolonları ortak boyutta bir token dizisine dönüştürmek."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Aşağıdaki denklemler bu dönüşümün matematiksel karşılığını vermektedir. Okuyucu açısından önemli olan sezgi şudur: her kolon, modelin "
            "okuyabildiği ortak boyutlu bir temsile çevrilir; sonra bu temsil dizisi self-attention katmanlarına verilir."
        )
    )
    builder.add_equation(r"T_j = b_j + f_j(x_j), \quad T_j \in \mathbb{R}^{d_{model}}", "Genel feature tokenizer tanımı.")
    builder.add_paragraph(
        ParagraphSpec(
            "Burada x_j ilgili özelliğin ham girdisini, f_j(.) ise bu girdiyi d_model boyutlu temsil uzayına taşıyan dönüşümü ifade etmektedir. "
            "Bu yapı sayesinde sayısal ve kategorik değişkenler, ortak bir token dizisi içinde birlikte işlenebilmektedir."
        )
    )
    builder.add_equation(r"T_j^{(num)} = b_j^{(num)} + x_j^{(num)} W_j^{(num)}", "Sayısal özellikler için token üretimi.")
    builder.add_equation(r"T_j^{(cat)} = b_j^{(cat)} + e_j^{T} W_j^{(cat)}", "Kategorik özellikler için token üretimi.")
    builder.add_equation(r"T = stack([T_1,\ldots,T_k]) \in \mathbb{R}^{k \times d_{model}}", "Token matrisinin oluşturulması.")
    builder.add_paragraph(
        ParagraphSpec(
            "Bu tokenizasyon yapısı, uçuş performans verilerinde önemli olan kolonlar arası ilişkilerin ortak bir temsil uzayında öğrenilmesini "
            "mümkün kılmaktadır. Başka bir ifadeyle model, yalnızca tek tek değişkenlerin etkisini değil; irtifa–Mach, gross_weight–drag_index "
            "ve fuel_flow–engine_type gibi etkileşimleri de encoder katmanları içinde birlikte işleyebilecek bir temele sahip olmaktadır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Pratik açıdan bakıldığında `x_j`, ilgili kolonun ham sayısal veya kategorik girdisidir; `f_j(.)` ise bu ham girdiyi modelin içinde çalışacağı "
            "ortak vektör uzayına taşıyan öğrenilebilir dönüşümdür. Yani tokenizer, veriyi yalnızca kopyalamaz; her kolonu modelin daha sonra dikkat "
            "mekanizmasıyla karşılaştırabileceği ortak bir formata çevirir."
        )
    )

    builder.add_paragraph(ParagraphSpec("6. Transformer Encoder Bloğu: MHSA, FFN, LayerNorm ve Residual Akışı", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Transformer encoder katmanının temelini, çok başlıklı öz-dikkat mekanizması (Multi-Head Self-Attention, MHSA) ile beslemeli "
            "ileri ağ (Feed-Forward Network, FFN) blokları oluşturmaktadır. Öz-dikkat mekanizmasının temel ifadesi aşağıdaki gibi tanımlanır."
        )
    )
    builder.add_equation(r"Attention(Q,K,V)=softmax\left(\frac{QK^{T}}{\sqrt{d_k}}\right)V", "Temel öz-dikkat ifadesi.")
    builder.add_paragraph(
        ParagraphSpec(
            "Bu ifadede `Q`, `K` ve `V` sırasıyla sorgu, anahtar ve değer temsilleridir. Sezgisel olarak model, her token için diğer tokenlara ne kadar "
            "bakması gerektiğini bu benzerlik yapısı üzerinden belirler. Böylece kolonlar arası etkileşim sabit bir kuralla değil, örneğe bağlı olarak öğrenilir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Çok başlıklı dikkat (multi-head) yapısı, bu işlemin tek bir bakış açısından değil birden fazla alt uzay üzerinden yapılmasını sağlar. Bir başlık "
            "irtifa–Mach ilişkisine daha duyarlı hale gelirken, başka bir başlık ağırlık–drag_index veya fuel_flow–engine_type ilişkisine daha fazla odaklanabilir. "
            "Bu da aynı satır içindeki farklı etkileşim tiplerinin paralel olarak işlenmesine imkân verir."
        )
    )
    builder.add_equation(r"FFN(z)=W_2 \sigma(W_1 z + b_1)+b_2", "Beslemeli ileri ağ bloğu.")
    builder.add_equation(r"Z' = Z + Dropout(MHSA(LN(Z)))", "Pre-norm encoder akışının ilk residual adımı.")
    builder.add_equation(r"Z_{new} = Z' + Dropout(FFN(LN(Z')))", "Pre-norm encoder akışının ikinci residual adımı.")
    builder.add_paragraph(
        ParagraphSpec(
            "Bu yapı, öğrenmenin daha kararlı hâle gelmesini sağlamakta ve derin katmanlarda bilgi akışının korunmasına yardımcı olmaktadır. "
            "Residual bağlantılar, giriş temsilinin tamamen kaybolmasını önlerken; LayerNorm, katmanlar boyunca sayısal ölçeğin dengelenmesine "
            "katkı sunmaktadır. MHSA bloğu yaklaşık O(T²) ölçeğinde dikkat matrisi maliyeti taşırken, FFN bloğunda yük daha çok d_ff ve "
            "d_model ile ölçeklenen matris çarpımlarında yoğunlaşmaktadır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Bu bölümün mühendislik açısından özeti şudur: MHSA ilişkileri kurar, FFN bu ilişkileri doğrusal olmayan biçimde zenginleştirir, "
            "LayerNorm sayısal ölçeği dengeler ve residual bağlantılar bilginin kaybolmasını önler. Bu dört yapı birlikte, FT-Transformer encoder "
            "bloğunun hem temsil gücünü hem de eğitim kararlılığını sağlar."
        )
    )
    builder.add_image(
        ImageSpec(
            report_figures["encoder_block"],
            "Transformer encoder blogunun sadelestirilmis akis semasi. Sema, Vaswani vd. (2017) encoder mantigi ile FT-Transformer uygulamasinin birlestirilmis yorumudur.",
            6.5,
        )
    )

    builder.add_paragraph(ParagraphSpec("7. Hiperparametreler ve Donanım Kısıtları", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "FT-Transformer mimarisinde ayarlanan temel hiperparametreler; encoder katman sayısı L, dikkat başlığı sayısı h, model boyutu "
            "d_model, FFN genişliği d_ff, dropout oranı ve öğrenme oranı olarak tanımlanmıştır. Bu parametreler yalnızca model doğruluğunu "
            "değil, aynı zamanda çıkarım süresi ve model boyutunu da etkilediğinden, seçim süreci doğruluk–hız–boyut ödünleşimi altında "
            "değerlendirilmiştir."
        )
    )
    builder.add_equation(r"P_{block} \approx 4d_{model}^{2} + 2d_{model}d_{ff}", "Bir encoder bloğu için yaklaşık parametre sayısı.")
    builder.add_equation(r"P_{total} \approx L \cdot P_{block} + P_{tokenizer} + P_{head}", "Toplam model parametre sayısının yaklaşık ifadesi.")
    builder.add_paragraph(
        ParagraphSpec(
            "Bu yaklaşım, daha büyük modellerin daha yüksek temsil gücü sunsa da edge hedefleri altında her zaman en uygun çözüm olmayabileceğini "
            "göstermektedir. Model tasarımında yalnızca doğruluk artışı değil, aynı zamanda NVIDIA Jetson sınıfı edge platformlarda yönetilebilir "
            "çıkarım gecikmesi, bellek kullanımı ve model boyutu da hedeflenmiştir. Bu nedenle ONNX ve TensorRT tabanlı dağıtım hattı ile INT8 "
            "nicemleme seçenekleri tasarımın doğal uzantısı olarak ele alınmıştır."
        )
    )

    builder.add_paragraph(ParagraphSpec("8. PSO ile Hiperparametre Optimizasyonu", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "FT-Transformer hiperparametre uzayı ayrık ve sürekli değişkenleri birlikte içerdiğinden, klasik grid arama yaklaşımı yerine türev "
            "gerektirmeyen ve çok amaçlı optimizasyona uygun bir yöntem tercih edilmiştir. Bu bağlamda Particle Swarm Optimization (PSO), "
            "doğruluk ile birlikte gecikme ve boyut hedeflerini de dikkate alan arama mekanizması olarak kullanılmıştır."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "PSO’da her parçacık bir hiperparametre vektörünü temsil etmekte, parçacıklar arama uzayında hem kendi geçmiş en iyi konumlarına "
            "hem de sürünün en iyi çözümüne göre güncellenmektedir. Ayrık hiperparametreler için projeksiyon yaklaşımı kullanılmış; sürekli "
            "uzayda güncellenen değerler her iterasyonda geçerli ayrık kümelere yuvarlanarak mimari tutarlılık korunmuştur."
        )
    )
    builder.add_equation(
        r"J(\theta)=w_1 \frac{RMSE(\theta)}{RMSE_{ref}}+w_2 \frac{T_{inf}(\theta)}{T_{ref}}+w_3 \frac{S(\theta)}{S_{ref}}",
        "Çok amaçlı PSO maliyet fonksiyonu.",
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Burada RMSE(theta) doğruluk bileşenini, T_inf(theta) çıkarım süresini ve S(theta) model boyutunu temsil etmektedir. Böylece PSO "
            "araması, yalnızca en düşük hatayı veren modeli değil; aynı zamanda gecikme ve boyut açısından daha dengeli çözümleri de "
            "değerlendirebilmektedir."
        )
    )

    builder.add_paragraph(ParagraphSpec("9. PSO Akışı, Deney Tasarımı ve Kıyaslama", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "PSO tabanlı hiperparametre optimizasyonu, bu çalışmada eğitim, doğrulama ve edge ölçüm adımlarını birlikte içeren uçtan uca bir "
            "döngü olarak ele alınmıştır. Her iterasyonda parçacıklar geçerli hiperparametre uzayına projekte edilmekte, ilgili FT-Transformer "
            "modeli eğitim verisi üzerinde eğitilmekte ve doğrulama kümesinde hata metriği hesaplanmaktadır. Ardından model ONNX formatına "
            "aktarılmakta, TensorRT engine oluşturulmakta ve hedef donanım üzerinde çıkarım süresi, tepe bellek kullanımı ve model boyutu "
            "ölçülmektedir. Bu ölçümler üzerinden maliyet fonksiyonu hesaplanmakta, parçacıkların en iyi konumları güncellenmekte ve süreç "
            "yinelemeli olarak sürdürülmektedir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Adil kıyas için XGBoost ve FT-Transformer modellerinin aynı deney protokolü altında karşılaştırılması esas alınmıştır. Bu kapsamda "
            "her iki model için aynı train/validation/test bölmesi, aynı ön işleme adımları, aynı aykırı değer yönetimi ve aynı donanım ölçüm "
            "prosedürü kullanılmalıdır. Hiperparametre araması yalnızca validation sonuçlarıyla yönlendirilmeli, test kümesi ise yalnızca nihai "
            "performans raporlamasında kullanılmalıdır. Böylece model karşılaştırmasının, protokol farklarından değil doğrudan yöntem farklarından "
            "kaynaklanması sağlanmaktadır."
        )
    )
    builder.add_image(
        ImageSpec(
            report_figures["comparison_protocol"],
            "XGBoost ve FT-Transformer icin adil karsilastirma protokolu. Iki model de ayni on isleme ve ayni metriklerle degerlendirilir.",
            6.4,
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Ön işleme aşamasında ölçekleme ve dönüşüm adımları yalnızca eğitim verisi üzerinde fit edilerek validation ve test kümelerine "
            "uygulanmalıdır. Aykırı değer etkisini azaltmak için Fuel Flow ve Specific Range değişkenlerinde p01–p99 aralığına dayalı kırpma "
            "uygulanabilir. Performans değerlendirmesinde RMSE, MAE, R² ve MAPE metrikleri birlikte raporlanmalı; edge tarafında ise çıkarım "
            "süresi, tepe bellek kullanımı ve model boyutu dikkate alınmalıdır."
        )
    )
    builder.add_equation(r"RMSE=\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2}", "Kök ortalama kare hata (RMSE).")
    builder.add_equation(r"MAE=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat{y}_i|", "Ortalama mutlak hata (MAE).")
    builder.add_equation(r"R^{2}=1-\frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}", "Belirlilik katsayısı.")
    builder.add_equation(r"MAPE=\frac{1}{n}\sum_{i=1}^{n}\left|\frac{y_i-\hat{y}_i}{y_i}\right|", "Ortalama mutlak yüzde hata (MAPE).")

    builder.add_paragraph(ParagraphSpec("10. Mevcut Kod Tabanından Örnek Sonuçlar", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Aşağıdaki tablolar ve şekiller, proje kapsamında üretilen mevcut eğitim ve raporlama çıktılarından seçilmiştir. Amaç, raporda hem "
            "yöntemsel çerçeveyi hem de kod tabanının somut çıktılarını birlikte gösterebilmektir."
        )
    )
    builder.add_paragraph(ParagraphSpec("Doğrulama ve test metrikleri", style="Heading2"))
    builder.add_table(TableSpec(build_metrics_table(root), col_widths=[1800, 1500, 1500, 1500, 1500, 1500]))
    builder.add_paragraph(ParagraphSpec("Toplu satır-bazlı rapor özeti", style="Heading2"))
    builder.add_table(TableSpec(build_row_level_table(root), col_widths=[1800, 1200, 1400, 1400, 1400, 1400, 1700]))

    xgb_plot = root / "artifacts" / "xgboost" / "reports" / "xgboost_slice_predictions.png"
    ft_plot = root / "artifacts" / "ft_transformer" / "reports" / "ft_transformer_slice_predictions.png"
    xgb_summary_plot = root / "artifacts" / "xgboost" / "reports" / "xgboost_slice_summary.png"
    ft_nomogram = find_first_existing([
        root / "artifacts" / "ft_transformer" / "nomogram_reports" / "ft_transformer_nomogram_one_engine_5000ft_50000lb.png",
        root / "artifacts" / "ft_transformer" / "nomogram_reports" / "ft_transformer_nomogram_one_engine_5000ft_30000lb.png",
    ])
    if xgb_plot.exists():
        builder.add_image(ImageSpec(xgb_plot, "XGBoost için gerçek ve tahmin eğrilerinin slice düzeyinde karşılaştırılması.", 6.3))
    if ft_plot.exists():
        builder.add_image(ImageSpec(ft_plot, "FT-Transformer için gerçek ve tahmin eğrilerinin slice düzeyinde karşılaştırılması.", 6.3))
    if xgb_summary_plot.exists():
        builder.add_image(ImageSpec(xgb_summary_plot, "XGBoost için slice düzeyi hata özeti.", 6.0))
    if ft_nomogram is not None:
        builder.add_image(ImageSpec(ft_nomogram, "FT-Transformer için taslak nomogram karşılaştırması örneği.", 6.0))

    builder.add_paragraph(ParagraphSpec("11. Sonuç ve Değerlendirme", style="Heading1"))
    builder.add_paragraph(
        ParagraphSpec(
            "Bu rapor, çoklu irtifa içeren uçak performans verilerinde FT-Transformer kullanımının neden anlamlı bir aday olduğunu hem veri yapısı hem "
            "de mimari bileşenler üzerinden açıklamaktadır. Problem yalnızca bir tablo satırını tahmin etmek değil, altitude, gross_weight, drag_index, "
            "mach, fuel_flow ve engine_type değişkenlerinin oluşturduğu çok boyutlu performans yüzeyini modellemektir."
        )
    )
    builder.add_paragraph(
        ParagraphSpec(
            "Mevcut kod tabanı, veri temizleme, eğitim, kıyaslama, full-table raporlama ve taslak nomogram üretimi gibi pratik araştırma adımlarını "
            "çalışır biçimde sunmaktadır. Bu nedenle rapor, yalnızca kuramsal bir mimari tartışması değil; aynı zamanda gerçek veri üzerinde denenmiş bir "
            "araştırma hattının belgelenmiş özetidir. Sonraki aşamada, görsellerin daha da zenginleştirilmesi, deployment ölçümlerinin tamamlanması ve "
            "Jetson/TensorRT tarafındaki benchmark çıktılarının eklenmesiyle rapor daha da güçlendirilebilir."
        )
    )

    builder.add_paragraph(ParagraphSpec("12. Taslak Kaynakça", style="Heading1"))
    for reference in [
        "[1] Y. Gorishniy, I. Rubachev, V. Khrulkov ve A. Babenko, “Revisiting Deep Learning Models for Tabular Data,” Advances in Neural Information Processing Systems, 2021.",
        "[2] T. Chen ve C. Guestrin, “XGBoost: A Scalable Tree Boosting System,” Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, ss. 785–794, 2016.",
        "[3] J. Kennedy ve R. Eberhart, “Particle Swarm Optimization,” Proceedings of ICNN’95 – International Conference on Neural Networks, c. 4, ss. 1942–1948, 1995.",
        "[4] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser ve I. Polosukhin, “Attention Is All You Need,” Advances in Neural Information Processing Systems, 2017.",
        "[5] X. Huang, A. Khetan, M. Cvitkovic ve Z. Karnin, “TabTransformer: Tabular Data Modeling Using Contextual Embeddings,” arXiv preprint arXiv:2012.06678, 2020.",
        "[6] F/A-18E/F Flight Manual: Performance Data for Pilots (A1-F18EA-NFM-200).",
        "[7] A. Sadou ve E. Njoya, “Applications of Artificial Intelligence in the Air Transport Industry,” Journal of Aerospace Technology and Management, 2022.",
        "[8] L. Grinsztajn, E. Oyallon ve G. Varoquaux, “Why Do Tree-Based Models Still Outperform Deep Learning on Typical Tabular Data?” NeurIPS 2022 Datasets and Benchmarks, 2022.",
        "[9] NVIDIA, Jetson Benchmarks and Deep Learning Latency Optimization, NVIDIA Documentation, 2023.",
        "[10] NVIDIA, TensorRT Developer Guide, NVIDIA Documentation.",
    ]:
        builder.add_paragraph(ParagraphSpec(reference))
    return builder


def build_outputs(root: Path) -> list[Path]:
    builder = build_report(root)
    write_gap_matrix(root, builder)
    requested_outputs = [root / "Rapor_Taslagi_Tabular_Transformer.docx", root / "Rapor_Taslagi_Tabular_Transformer_Akademik.docx"]
    written_outputs: list[Path] = []
    for output in requested_outputs:
        try:
            write_docx(output, builder)
            written_outputs.append(output)
        except PermissionError:
            fallback = output.with_stem(output.stem + "_yeni")
            write_docx(fallback, builder)
            written_outputs.append(fallback)
    return written_outputs


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    for output_path in build_outputs(project_root):
        print(output_path)
