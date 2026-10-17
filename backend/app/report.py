"""Professional PDF report rendering with ReportLab.

Layout follows a radiology report: institution header, patient demographics, clinical
indication, highlighted impression, images, findings tables, notes, sign-off block, and a
footer with "Page X of Y" on every page.
"""

import base64
import binascii
import io
import uuid
from datetime import date, datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

MAX_IMAGE_BYTES = 4 * 1024 * 1024

DISCLAIMER = (
    "Research and education only. NeuroScan AI is not a medical device and is not cleared for clinical use. "
    "AI output must be reviewed by a qualified clinician and must not be used to diagnose, treat or triage any patient."
)

INK = colors.HexColor("#0F1B2D")
MUTED = colors.HexColor("#5B6B80")
ACCENT = colors.HexColor("#1F57D6")
ACCENT_SOFT = colors.HexColor("#EAF0FD")
RULE = colors.HexColor("#CDD6E3")
ZEBRA = colors.HexColor("#F5F7FA")


class ReportSection(BaseModel):
    heading: str = Field(max_length=120)
    rows: list[tuple[str, str]] = Field(default_factory=list, max_length=60)


class ReportImage(BaseModel):
    caption: str = Field(max_length=160)
    data_url: str

    @field_validator("data_url")
    @classmethod
    def _decode_check(cls, v: str) -> str:
        if not v.startswith(("data:image/png;base64,", "data:image/jpeg;base64,")):
            raise ValueError("image must be a PNG or JPEG data URL")
        return v

    def raw(self) -> bytes:
        try:
            data = base64.b64decode(self.data_url.split(",", 1)[1], validate=True)
        except (binascii.Error, IndexError) as exc:
            raise ValueError("invalid base64 image") from exc
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("image too large")
        return data


class PatientInfo(BaseModel):
    name: str = Field(default="", max_length=120)
    patient_id: str = Field(default="", max_length=60)
    age: int | None = Field(default=None, ge=0, le=130)
    sex: Literal["", "female", "male", "other"] = ""
    date_of_birth: date | None = None
    scan_date: date | None = None
    referring_physician: str = Field(default="", max_length=120)
    institution: str = Field(default="", max_length=120)
    clinical_indication: str = Field(default="", max_length=1000)


class ReportRequest(BaseModel):
    title: str = Field(max_length=120)
    study_id: str = Field(default="", max_length=80)
    analyst: str = Field(default="", max_length=80)
    notes: str = Field(default="", max_length=2000)
    modality: str = Field(default="", max_length=120)
    impression: str = Field(default="", max_length=1200)
    patient: PatientInfo = Field(default_factory=PatientInfo)
    sections: list[ReportSection] = Field(default_factory=list, max_length=12)
    images: list[ReportImage] = Field(default_factory=list, max_length=6)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    p = lambda name, **kw: ParagraphStyle(name, parent=base["Normal"], **{"fontName": "Helvetica", "textColor": INK, **kw})
    return {
        "title": p("title", fontName="Helvetica-Bold", fontSize=17, leading=21),
        "kicker": p("kicker", fontName="Helvetica-Bold", fontSize=7.5, leading=10, textColor=ACCENT),
        "h2": p("h2", fontName="Helvetica-Bold", fontSize=10.5, leading=13, spaceBefore=12, spaceAfter=6, textColor=ACCENT),
        "body": p("body", fontSize=9.5, leading=13.5),
        "small": p("small", fontSize=8, leading=10.5, textColor=MUTED),
        "label": p("label", fontName="Helvetica-Bold", fontSize=7, leading=9, textColor=MUTED),
        "value": p("value", fontSize=9.5, leading=12),
        "impression": p("impression", fontName="Helvetica-Bold", fontSize=10.5, leading=15),
    }


def _esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _or_dash(value: object) -> str:
    return _esc(str(value)) if value not in (None, "") else "—"


def _section_title(text: str, st: dict) -> Paragraph:
    return Paragraph(_esc(text).upper(), st["h2"])


def _kv_table(rows: list[tuple[str, str]], st: dict, widths: tuple[float, float]) -> Table:
    data = [[Paragraph(_esc(k), st["small"]), Paragraph(_esc(v) or "—", st["value"])] for k, v in rows]
    table = Table(data, colWidths=widths, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, ZEBRA]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("TOPPADDING", (0, 0), (-1, -1), 4.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    return table


def _patient_box(req: ReportRequest, st: dict, width: float, report_id: str, generated: str) -> Table:
    pt = req.patient
    age_sex = " / ".join(x for x in [f"{pt.age} y" if pt.age is not None else "", pt.sex.capitalize() if pt.sex else ""] if x)
    cells = [
        ("PATIENT NAME", _or_dash(pt.name)),
        ("PATIENT ID", _or_dash(pt.patient_id)),
        ("AGE / SEX", _or_dash(age_sex)),
        ("DATE OF BIRTH", _or_dash(pt.date_of_birth.isoformat() if pt.date_of_birth else "")),
        ("SCAN DATE", _or_dash(pt.scan_date.isoformat() if pt.scan_date else "")),
        ("REFERRING PHYSICIAN", _or_dash(pt.referring_physician)),
        ("ACCESSION / STUDY ID", _or_dash(req.study_id)),
        ("MODALITY / STUDY", _or_dash(req.modality)),
        ("REPORT ID", report_id),
        ("REPORT GENERATED", generated),
    ]
    rows = []
    for i in range(0, len(cells), 3):
        chunk = cells[i:i + 3] + [("", "")] * (3 - len(cells[i:i + 3]))
        rows.append([[Paragraph(label, st["label"]), Paragraph(value, st["value"])] for label, value in chunk])
    col = width / 3
    table = Table(rows, colWidths=(col, col, col), hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    return table


def _impression_box(text: str, st: dict, width: float) -> Table:
    box = Table(
        [[Paragraph("IMPRESSION", st["label"])], [Paragraph(_esc(text).replace("\n", "<br/>"), st["impression"])]],
        colWidths=(width,), hAlign="LEFT",
    )
    box.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
        ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (0, 0), 8),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
    ]))
    return box


def _signoff(req: ReportRequest, st: dict, width: float, generated: str) -> Table:
    left = [
        Paragraph("ANALYSED BY", st["label"]),
        Paragraph(_esc(req.analyst) or "NeuroScan AI (automated analysis)", st["value"]),
        Paragraph(f"Generated {generated}", st["small"]),
    ]
    right = [
        Paragraph("REVIEWED BY (CLINICIAN)", st["label"]),
        Spacer(1, 18),
        Paragraph("Signature ______________________________", st["value"]),
        Paragraph("Name / date ____________________________", st["value"]),
    ]
    t = Table([[left, right]], colWidths=(width / 2, width / 2), hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, RULE),
        ("LINEBEFORE", (1, 0), (1, 0), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
    ]))
    return t


def _image_grid(images: list[ReportImage], st: dict, width: float) -> Table:
    cell_w = (width - 6 * mm) / 2
    cells = []
    for item in images:
        raw = item.raw()
        iw, ih = ImageReader(io.BytesIO(raw)).getSize()
        scale = min(cell_w / iw, 68 * mm / ih)
        cells.append([Image(io.BytesIO(raw), width=iw * scale, height=ih * scale), Paragraph(_esc(item.caption), st["small"])])
    rows = []
    for i in range(0, len(cells), 2):
        pair = cells[i:i + 2] + ([["", ""]] if len(cells[i:i + 2]) == 1 else [])
        rows.append([pair[0][0], pair[1][0]])
        rows.append([pair[0][1], pair[1][1]])
    grid = Table(rows, colWidths=(cell_w + 3 * mm, cell_w + 3 * mm), hAlign="LEFT")
    grid.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return grid


def render_report(req: ReportRequest) -> tuple[bytes, str]:
    report_id = uuid.uuid4().hex[:10].upper()
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    st = _styles()
    buf = io.BytesIO()
    page_w, page_h = A4
    margin = 16 * mm
    content_w = page_w - 2 * margin
    institution = req.patient.institution.strip() or "NeuroScan AI · Neuro-oncology Research"

    class NumberedCanvas(rl_canvas.Canvas):
        """Two-pass canvas so every page can show 'Page X of Y'."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pages: list[dict] = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._pages)
            for state in self._pages:
                self.__dict__.update(state)
                self._decorate(total)
                super().showPage()
            super().save()

        def _decorate(self, total: int):
            self.saveState()
            # Header band
            self.setFillColor(INK)
            self.rect(0, page_h - 22 * mm, page_w, 22 * mm, stroke=0, fill=1)
            self.setFillColor(ACCENT)
            self.rect(0, page_h - 23.2 * mm, page_w, 1.2 * mm, stroke=0, fill=1)
            self.setFillColor(colors.white)
            self.setFont("Helvetica-Bold", 12)
            self.drawString(margin, page_h - 11 * mm, _esc(institution)[:70])
            self.setFont("Helvetica", 8)
            self.setFillColor(colors.HexColor("#B8C4D6"))
            self.drawString(margin, page_h - 16 * mm, "AI-assisted brain MRI analysis report")
            self.drawRightString(page_w - margin, page_h - 11 * mm, f"Report {report_id}")
            self.drawRightString(page_w - margin, page_h - 16 * mm, generated)
            # Footer
            self.setStrokeColor(RULE)
            self.line(margin, 15 * mm, page_w - margin, 15 * mm)
            self.setFont("Helvetica", 7)
            self.setFillColor(MUTED)
            self.drawString(margin, 10.5 * mm, "CONFIDENTIAL: contains patient information. Research use only, not for clinical decisions.")
            self.drawRightString(page_w - margin, 10.5 * mm, f"Page {self._pageNumber} of {total}")
            self.restoreState()

    doc = SimpleDocTemplate(
        buf, pagesize=A4, leftMargin=margin, rightMargin=margin, topMargin=30 * mm, bottomMargin=22 * mm,
        title=f"{req.title} · {report_id}", author="NeuroScan AI", subject="AI-assisted brain MRI analysis",
    )

    story: list = [
        Paragraph("DIAGNOSTIC IMAGING REPORT · AI-ASSISTED", st["kicker"]),
        Spacer(1, 2),
        Paragraph(_esc(req.title), st["title"]),
        Spacer(1, 8),
        _patient_box(req, st, content_w, report_id, generated),
    ]

    if req.patient.clinical_indication.strip():
        story += [_section_title("Clinical indication", st),
                  Paragraph(_esc(req.patient.clinical_indication).replace("\n", "<br/>"), st["body"])]

    if req.impression.strip():
        story += [Spacer(1, 10), _impression_box(req.impression, st, content_w)]

    if req.images:
        story += [_section_title("Images", st), _image_grid(req.images, st, content_w)]

    if req.sections:
        story.append(_section_title("Findings", st))
        for section in req.sections:
            story.append(KeepTogether([
                Paragraph(_esc(section.heading), ParagraphStyle("sh", parent=st["body"], fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=4)),
                _kv_table(list(section.rows), st, (64 * mm, content_w - 64 * mm)),
            ]))

    if req.notes.strip():
        story += [_section_title("Notes", st), Paragraph(_esc(req.notes).replace("\n", "<br/>"), st["body"])]

    story += [
        Spacer(1, 14),
        KeepTogether([_section_title("Sign-off", st), _signoff(req, st, content_w, generated)]),
        Spacer(1, 10),
        Paragraph(DISCLAIMER, st["small"]),
    ]

    doc.build(story, canvasmaker=NumberedCanvas)
    return buf.getvalue(), report_id
