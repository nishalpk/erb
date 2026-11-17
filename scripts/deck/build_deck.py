"""Build the ERB2 final-presentation deck on the BITS template.

    python build_deck.py <template.pptx> <ERB2 repo root> <deckfigs dir> <out.pptx>
"""
import copy
import sys
from pathlib import Path

import pandas as pd
from lxml import etree
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

TPL, ROOT, FIGS, OUT = (Path(a) for a in sys.argv[1:5])
TAB = ROOT / "reports" / "tables"
IMG = ROOT / "data" / "processed"

# ---------------------------------------------------------------- palette (from the BITS template)
NAVY = RGBColor(0x10, 0x11, 0x41)
TITLE = RGBColor(0x00, 0x20, 0x60)
ORANGE = RGBColor(0xFC, 0xB0, 0x17)
DORANGE = RGBColor(0xE3, 0x9B, 0x00)
SKY = RGBColor(0x76, 0xC2, 0xE5)
RED = RGBColor(0xE3, 0x1C, 0x24)
BLUE = RGBColor(0x1F, 0x6F, 0xB5)
INK = RGBColor(0x1A, 0x1A, 0x1A)
GREY = RGBColor(0x5F, 0x5F, 0x5F)
LGREY = RGBColor(0xD9, 0xD9, 0xD9)
TINT = RGBColor(0xEE, 0xF5, 0xFB)   # light sky tint for cards
OTINT = RGBColor(0xFF, 0xF4, 0xDC)  # light orange tint
RTINT = RGBColor(0xFC, 0xE4, 0xE4)  # light red tint
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MODELS = [("smolvlm2-2.2b", "SmolVLM2-2.2B", DORANGE), ("qwen2.5-vl-3b", "Qwen2.5-VL-3B", BLUE),
          ("llava-ov-0.5b", "LLaVA-OV-0.5B", RED)]
MNAME = {k: n for k, n, _ in MODELS}
CONDS = ["none", "neutral", "consistent", "contra_1", "contra_2", "contra_3", "contra_4"]
CLAB = ["none", "neutral", "true", "hedged", "plain", "\"definitely\"", "\"verified\""]
FONT = "Calibri"

prs = Presentation(str(TPL))
LAYOUT = next(l for l in prs.slide_layouts if l.name == "Title and Content")
X0, W = 0.45, 9.1


# ---------------------------------------------------------------- helpers
def hexs(c):
    return str(c)


def drop_slide(idx):
    sldIdLst = prs.slides._sldIdLst
    sid = sldIdLst[idx]
    prs.part.drop_rel(sid.rId)
    sldIdLst.remove(sid)


def set_run(r, text, size=14, bold=False, color=INK, italic=False, font=FONT):
    r.text = text
    f = r.font
    f.size, f.bold, f.italic, f.name = Pt(size), bold, italic, font
    f.color.rgb = color


def tb(slide, x, y, w, h, paras, anchor=MSO_ANCHOR.TOP, margin=0.05, name=None, autofit=False):
    """paras: list of paragraphs; each a list of (text, opts) runs or a dict {runs, align, bullet, space}"""
    s = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    tf = s.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    fill_tf(tf, paras)
    return s


def fill_tf(tf, paras):
    first = True
    for p in paras:
        if not isinstance(p, dict):
            p = {"runs": p}
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        para.alignment = p.get("align", PP_ALIGN.LEFT)
        if p.get("space") is not None:
            para.space_after = Pt(p["space"])
        if p.get("before") is not None:
            para.space_before = Pt(p["before"])
        if p.get("bullet"):
            pPr = para._p.get_or_add_pPr()
            ind = p.get("indent", 0.22)
            pPr.set("marL", str(int(Inches(ind))))
            pPr.set("indent", str(-int(Inches(ind))))
            bu = etree.SubElement(pPr, qn("a:buFont"))
            bu.set("typeface", "Arial")
            bc = etree.SubElement(pPr, qn("a:buChar"))
            bc.set("char", p["bullet"] if isinstance(p["bullet"], str) else "•")
        for text, o in p["runs"]:
            r = para.add_run()
            set_run(r, text, **o)


def rect(slide, x, y, w, h, fill=TINT, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, name=None, lw=1.0):
    s = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    if name:
        s.name = name
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(lw)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    s.shadow.inherit = False
    tf = s.text_frame
    tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.1))
    return s


def card(slide, x, y, w, h, paras, fill=TINT, line=None, anchor=MSO_ANCHOR.TOP, name=None, margin=0.12):
    s = rect(slide, x, y, w, h, fill=fill, line=line, name=name)
    tf = s.text_frame
    tf.vertical_anchor = anchor
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(margin))
    fill_tf(tf, paras)
    return s


def chip(slide, x, y, d, text, fill=NAVY, color=WHITE, size=16):
    s = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    s.line.fill.background()
    s.shadow.inherit = False
    tf = s.text_frame
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", 0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    fill_tf(tf, [{"runs": [(text, dict(size=size, bold=True, color=color))], "align": PP_ALIGN.CENTER}])
    return s


def arrow(slide, x1, y1, x2, y2, color=GREY, w=2.0):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    c.line.color.rgb = color
    c.line.width = Pt(w)
    ln = c.line._get_or_add_ln()
    tail = etree.SubElement(ln, qn("a:tailEnd"))
    tail.set("type", "triangle")
    tail.set("w", "med")
    tail.set("len", "med")
    return c


def picture(slide, path, x, y, w=None, h=None, border=None):
    p = slide.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)
    if border is not None:
        p.line.color.rgb = border
        p.line.width = Pt(0.75)
    return p


# ---------------------------------------------------------------- native PowerPoint equations (OMML)
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
A14_NS = "http://schemas.microsoft.com/office/drawing/2010/main"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"


def _esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def R(t, it=None):
    """math run; single letters are italic (math convention), words upright"""
    if it is None:
        it = len(t) == 1 and t.isalpha()
    sty = "" if it else '<m:rPr><m:sty m:val="p"/></m:rPr>'
    return f'<m:r>{sty}@RPR{int(it)}@<m:t xml:space="preserve">{_esc(t)}</m:t></m:r>'


def F(n, d):
    return f"<m:f><m:num>{n}</m:num><m:den>{d}</m:den></m:f>"


def SUB(b, s):
    return f"<m:sSub><m:e>{b}</m:e><m:sub>{s}</m:sub></m:sSub>"


def SUP(b, s):
    return f"<m:sSup><m:e>{b}</m:e><m:sup>{s}</m:sup></m:sSup>"


def SUM(sub, e):
    return (f'<m:nary><m:naryPr><m:chr m:val="∑"/><m:limLoc m:val="undOvr"/><m:supHide m:val="1"/></m:naryPr>'
            f"<m:sub>{sub}</m:sub><m:sup/><m:e>{e}</m:e></m:nary>")


def D(e, l="(", r=")"):
    return f'<m:d><m:dPr><m:begChr m:val="{_esc(l)}"/><m:endChr m:val="{_esc(r)}"/></m:dPr><m:e>{e}</m:e></m:d>'


def equation(slide, x, y, w, h, omml, fallback, size=20, color=NAVY, align="center"):
    """Insert a real PowerPoint equation (editable in PowerPoint's equation editor) with a plain-text fallback."""
    sp = tb(slide, x, y, w, h, [{"runs": [(fallback, dict(size=size, color=color, font="Cambria Math"))],
                                 "align": PP_ALIGN.CENTER if align == "center" else PP_ALIGN.LEFT}],
            anchor=MSO_ANCHOR.MIDDLE, name="Equation")
    el = sp._element
    choice_sp = copy.deepcopy(el)
    rpr = {k: (f'<a:rPr lang="en-US" sz="{int(size * 100)}" b="0" i="{k}"><a:solidFill><a:srgbClr val="{color}"/>'
               f'</a:solidFill><a:latin typeface="Cambria Math"/><a:cs typeface="Cambria Math"/></a:rPr>') for k in (0, 1)}
    body = omml.replace("@RPR0@", rpr[0]).replace("@RPR1@", rpr[1])
    jc = "center" if align == "center" else "left"
    m_xml = (f'<a14:m xmlns:a14="{A14_NS}" xmlns:m="{M_NS}" xmlns:a="{A_NS}"><m:oMathPara><m:oMathParaPr>'
             f'<m:jc m:val="{jc}"/></m:oMathParaPr><m:oMath>{body}</m:oMath></m:oMathPara></a14:m>')
    p = choice_sp.find(".//" + qn("a:p"))
    for r_ in p.findall(qn("a:r")):
        p.remove(r_)
    p.insert(len(p.findall(qn("a:pPr"))), etree.fromstring(m_xml))
    ac = etree.Element(f"{{{MC_NS}}}AlternateContent", nsmap={"mc": MC_NS})
    ch_ = etree.SubElement(ac, f"{{{MC_NS}}}Choice", nsmap={"a14": A14_NS})
    ch_.set("Requires", "a14")
    ch_.append(choice_sp)
    fb = etree.SubElement(ac, f"{{{MC_NS}}}Fallback")
    el.addprevious(ac)
    fb.append(el)
    return ac


def new_slide(title, headline=None, notes=None):
    s = prs.slides.add_slide(LAYOUT)
    body, ttl = None, None
    for ph in s.placeholders:
        if ph.placeholder_format.idx == 10:
            ttl = ph
        elif ph.placeholder_format.idx == 1:
            body = ph
    ttl.text_frame.text = ""
    r = ttl.text_frame.paragraphs[0].add_run()
    r.text = title
    r.font.size = Pt(30)
    r.font.color.rgb = TITLE
    if body is not None:
        body._element.getparent().remove(body._element)
    if headline:
        tb(s, X0, 1.52, W, 0.5, [[(headline, dict(size=17, italic=True, color=NAVY, bold=True))]], anchor=MSO_ANCHOR.MIDDLE,
           name="Headline")
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def style_chart(ch, legend=True, size=11, legend_pos=XL_LEGEND_POSITION.BOTTOM):
    ch.font.size = Pt(size)
    ch.font.name = FONT
    ch.font.color.rgb = GREY
    ch.has_legend = legend
    if legend:
        ch.legend.position = legend_pos
        ch.legend.include_in_layout = False
        ch.legend.font.size = Pt(size)


def axis_style(ax, fmt=None, maxv=None, minv=None, gridlines=True, size=11):
    ax.tick_labels.font.size = Pt(size)
    ax.tick_labels.font.color.rgb = GREY
    ax.format.line.color.rgb = LGREY
    if fmt:
        ax.tick_labels.number_format = fmt
        ax.tick_labels.number_format_is_linked = False
    if maxv is not None:
        ax.maximum_scale = maxv
    if minv is not None:
        ax.minimum_scale = minv
    if hasattr(ax, "has_major_gridlines"):
        ax.has_major_gridlines = gridlines
        if gridlines:
            ax.major_gridlines.format.line.color.rgb = RGBColor(0xEC, 0xEC, 0xEC)


def chart_title(ch, text, size=12):
    ch.has_title = True
    tf = ch.chart_title.text_frame
    tf.text = text
    for r in tf.paragraphs[0].runs:
        r.font.size, r.font.bold, r.font.color.rgb, r.font.name = Pt(size), True, NAVY, FONT


def table(slide, x, y, w, rows, colw, size=12, header_fill=NAVY, row_h=0.32, fills=None, bold_first=False):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = t._tbl.tblPr
    for k in ("bandRow", "firstRow"):
        tblPr.set(k, "0")
    st = tblPr.find(qn("a:tableStyleId"))
    if st is not None:
        st.text = "{5940675A-B579-460E-94D1-54222C63F5DA}"  # No Style, Table Grid
    for j, cw in enumerate(colw):
        t.columns[j].width = Inches(cw)
    for i in range(nr):
        t.rows[i].height = Inches(row_h)
        for j in range(nc):
            c = t.cell(i, j)
            val = rows[i][j]
            opts = {}
            if isinstance(val, tuple):
                val, opts = val
            c.margin_left = c.margin_right = Inches(0.06)
            c.margin_top = c.margin_bottom = Inches(0.03)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame
            tf.word_wrap = True
            tf.text = ""
            r = tf.paragraphs[0].add_run()
            hdr = i == 0
            set_run(r, str(val), size=opts.get("size", size), bold=opts.get("bold", hdr or (bold_first and j == 0)),
                    color=opts.get("color", WHITE if hdr else INK))
            tf.paragraphs[0].alignment = opts.get("align", PP_ALIGN.LEFT)
            c.fill.solid()
            if hdr:
                c.fill.fore_color.rgb = header_fill
            elif opts.get("fill") is not None:
                c.fill.fore_color.rgb = opts["fill"]
            else:
                c.fill.fore_color.rgb = WHITE if i % 2 else RGBColor(0xF4, 0xF7, 0xFB)
    # thin grey borders
    for i in range(nr):
        for j in range(nc):
            tcPr = t.cell(i, j)._tc.get_or_add_tcPr()
            for tag in ("a:lnL", "a:lnR", "a:lnT", "a:lnB"):
                ln = etree.SubElement(tcPr, qn(tag))
                ln.set("w", "6350")
                sf = etree.SubElement(ln, qn("a:solidFill"))
                c_ = etree.SubElement(sf, qn("a:srgbClr"))
                c_.set("val", "D9D9D9")
    return gt


def series_color(series, color, line=False, width=2.25):
    if line:
        series.format.line.color.rgb = color
        series.format.line.width = Pt(width)
        series.marker.format.fill.solid()
        series.marker.format.fill.fore_color.rgb = color
        series.marker.format.line.color.rgb = color
        series.marker.size = 6
        series.smooth = False
    else:
        series.format.fill.solid()
        series.format.fill.fore_color.rgb = color


# ---------------------------------------------------------------- data
rr = pd.read_csv(TAB / "rr.csv")
cap = pd.read_csv(TAB / "capability.csv")
h1 = pd.read_csv(TAB / "h1.csv")
tax = pd.read_csv(TAB / "taxonomy_items.csv")
probe = pd.read_csv(TAB / "probe_by_layer.csv")
abl = pd.read_csv(TAB / "robust_ablation.csv")
funnel = pd.read_csv(IMG / "real" / "funnel.csv")

# ================================================================ 1. title (template slide 1)
s1 = prs.slides[0]
for ph in s1.placeholders:
    tf = ph.text_frame
    if ph.placeholder_format.type is not None and "TITLE" in str(ph.placeholder_format.type):
        tf.text = ""
        p = tf.paragraphs[0]
        r = p.add_run()
        r.text = "WHEN CAPTIONS LIE"
        r.font.size = Pt(30)
        p2 = tf.add_paragraph()
        r2 = p2.add_run()
        r2.text = "Do vision-language models trust text over their own eyes?"
        r2.font.size = Pt(15)
        r2.font.bold = False
        r2._r.get_or_add_rPr().set("spc", "0")
        r._r.get_or_add_rPr().set("spc", "0")
    else:
        paras = tf.paragraphs
        paras[0].runs[0].text = "Team: [Name 1, Name 2, Name 3]"
        paras[0].runs[0].font.size = Pt(16)
        paras[1].runs[0].text = "ID Nos.: [ID 1, ID 2, ID 3]"
        paras[1].runs[0].font.size = Pt(16)
tb(s1, 0.25, 0.25, 4.6, 0.7, [[("CS F320 Foundations of Data Science", dict(size=14, bold=True, color=WHITE))],
                               [("Final project presentation · Sem I 2026–27", dict(size=12, color=WHITE))]],
   name="Course tag")
s1.notes_slide.notes_text_frame.text = (
    "Open: We tested whether small AI models that look at images believe what they see, or what a caption tells them. "
    "Spoiler: a single confident sentence is enough to make them name the wrong person.")

# drop template slides 2-7 (contents/points/conclusion/references are rebuilt below)
for _ in range(len(prs.slides) - 1):
    drop_slide(1)

# ================================================================ 3. motivation
s = new_slide("INTRODUCTION", "One confident sentence can make a model name the wrong person",
              notes="This is a real photo from the V-COCO dataset. Person A is throwing the frisbee. With no caption, "
                    "Qwen2.5-VL-3B gives the right person 87% probability. Add a plain false caption 'B is throwing the "
                    "frisbee' and it drops to 37%: the model now names B. The image never changed. Only the text did. "
                    "Vision-language models are used to caption datasets, write alt-text for blind users and moderate "
                    "content; the text they receive is often wrong or adversarial.")
picture(s, IMG / "real" / "images" / "vcoco_000000085007_marked.jpg", X0, 2.15, h=3.45, border=LGREY)
card(s, X0, 5.72, 3.25, 1.1, [[("Caption shown to the model:", dict(size=12, bold=True, color=NAVY))],
                               [("“B is throwing the frisbee.”", dict(size=14, italic=True, color=RED, bold=True))],
                               [("Ground truth: A is throwing it.", dict(size=12, color=GREY))]], fill=RTINT)
tb(s, 4.0, 2.1, 5.55, 0.62, [[("Qwen2.5-VL-3B is asked: ", dict(size=14, color=GREY)),
                              ("“Who is throwing the frisbee, A or B?”", dict(size=14, bold=True, color=NAVY))]])
cd = CategoryChartData()
cd.categories = ["no caption", "true caption", "\"I think B might be…\"", "\"B is …\"", "\"Verified: B is …\""]
vals = [0.87, 0.92, 0.57, 0.37, 0.39]
cd.add_series("P(correct person A)", vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(4.0), Inches(2.7), Inches(5.55), Inches(2.6), cd)
ch = gf.chart
style_chart(ch, legend=False)
chart_title(ch, "Model's probability for the correct person (A)", 12)
ch.plots[0].gap_width = 45
ch.plots[0].has_data_labels = True
dl = ch.plots[0].data_labels
dl.number_format, dl.number_format_is_linked, dl.position = "0%", False, XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold = Pt(12), True
ser = ch.plots[0].series[0]
for i, v in enumerate(vals):
    pt = ser.points[i]
    pt.format.fill.solid()
    pt.format.fill.fore_color.rgb = BLUE if v >= 0.5 else RED
ca = ch.category_axis
ca.reverse_order = True
axis_style(ca, gridlines=False, size=11)
va = ch.value_axis
axis_style(va, fmt="0%", maxv=1.0, minv=0.0, size=10, gridlines=False)
from pptx.enum.chart import XL_TICK_LABEL_POSITION
va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
va.format.line.fill.background()
why = [("Dataset auto-captioning", "wrong labels propagate"), ("Alt-text for blind users", "trust in what is described"),
       ("Moderation & robotics", "text can be wrong or adversarial")]
for k, (h, b) in enumerate(why):
    card(s, 4.0 + k * 1.88, 5.5, 1.79, 1.32, [[(h, dict(size=13, bold=True, color=NAVY))], [(b, dict(size=12, color=GREY))]],
         fill=TINT)
tb(s, 4.0, 5.22, 5.55, 0.3, [[("Why it matters", dict(size=12, bold=True, color=RED))]])

# ================================================================ 4. novelty
s = new_slide("NOVELTY", "We separate who does the action from whether it is seen, and measure it statistically",
              notes="Prior work shows VLMs lean on text, usually reported as one accuracy drop. That cannot tell apart "
                    "two very different failures: the model stops seeing the action, or it still sees it but attaches "
                    "it to the wrong person. Our design isolates the second, which we call a binding failure, and "
                    "treats it as a data-science problem: a curated dataset, pre-registered hypotheses, paired tests, "
                    "regression, Bayesian estimates and a look inside the model.")
card(s, X0, 2.15, 3.6, 4.7, [{"runs": [("What is already known", dict(size=17, bold=True, color=NAVY))], "space": 8},
                             {"runs": [("Vision-language models lean on language priors and can be misled by text in the prompt.", dict(size=14))], "bullet": True, "space": 8},
                             {"runs": [("Usually reported as a single accuracy drop on a question-answering benchmark.", dict(size=14))], "bullet": True, "space": 8},
                             {"runs": [("Open question: did the model stop ", dict(size=14)), ("seeing", dict(size=14, bold=True)),
                                       (" the action, or attach it to the ", dict(size=14)), ("wrong person", dict(size=14, bold=True)), ("?", dict(size=14))], "bullet": True}],
     fill=RGBColor(0xF2, 0xF2, 0xF2))
adds = [("Controlled benchmark", "two-person images built by data wrangling: 48,464 → 152 real + 800 synthetic"),
        ("Graded lies", "4 confidence levels of false caption → a dose-response curve"),
        ("Control questions", "Q2 (action seen?) and Q3 (lie noticed?) separate binding from perception"),
        ("Rigorous statistics", "pre-registered H1–H5; McNemar, GEE logistic regression, Bayesian posteriors"),
        ("Inside the model", "logistic probe, PCA and activation patching on hidden states")]
rect(s, 4.25, 2.15, 5.3, 4.7, fill=NAVY)
tb(s, 4.45, 2.25, 4.9, 0.45, [[("What this project adds", dict(size=17, bold=True, color=WHITE))]])
for k, (h, b) in enumerate(adds):
    y = 2.78 + k * 0.81
    chip(s, 4.45, y + 0.04, 0.45, str(k + 1), fill=ORANGE, color=NAVY, size=14)
    tb(s, 5.02, y - 0.02, 4.4, 0.8, [[(h + ": ", dict(size=14, bold=True, color=ORANGE)), (b, dict(size=13, color=WHITE))]])

# ================================================================ 5. objectives
s = new_slide("OBJECTIVES", "Five objectives, each tied to a hypothesis and a data-science method",
              notes="Each objective maps to one pre-registered hypothesis and to methods from this course: "
                    "probability distributions and confidence intervals, hypothesis testing, logistic regression, "
                    "Bayesian inference, PCA and supervised learning (the probe).")
rows = [["#", "Objective", "Hypothesis", "Data-science method"],
        [("O1", {"bold": True, "color": NAVY}), "Build a clean, balanced benchmark from noisy public data", "Data quality (CVC1–5)",
         "Data wrangling, EDA, inter-rater agreement (Fleiss κ)"],
        [("O2", {"bold": True, "color": NAVY}), "Test whether a false caption flips who does the action", "H1: contradiction raises the reassignment rate",
         "Paired McNemar test, Holm correction, Wilson CI (Binomial)"],
        [("O3", {"bold": True, "color": NAVY}), "Measure how the effect grows with caption confidence", "H2: dose-response",
         "Cochran–Armitage trend, GEE logistic regression, Beta-Binomial posterior"],
        [("O4", {"bold": True, "color": NAVY}), "Tell rebinding apart from perception loss and awareness", "H3: action still seen\nH4: noticing does not protect",
         "Conditional probabilities, odds ratios, Fisher's exact test"],
        [("O5", {"bold": True, "color": NAVY}), "Check whether the truth survives inside the model", "H5: true actor still decodable",
         "Logistic-regression probe, PCA, activation patching"]]
table(s, X0, 2.15, W, rows, [0.5, 3.0, 2.55, 3.05], size=13, row_h=0.78)

# ================================================================ 6. block diagram
s = new_slide("BLOCK DIAGRAM", "End-to-end pipeline: from 48,464 raw images to tested hypotheses",
              notes="Three stages. DATA: we start from public action-labelled photos (mainly V-COCO), apply validity "
                    "rules plus an RT-DETR detector, and end with 152 clean real images plus 800 synthetic scenes. "
                    "EXPERIMENT: a query generator crosses 7 caption conditions with 4 questions and 2 answer orders; "
                    "each model answers and we read the probability of A versus B directly from the logits. "
                    "ANALYSIS: statistics for H1–H4, a probe, PCA and patching for H5, and the outputs: figures, "
                    "verdicts and the live demo.")
stages = [("DATA", NAVY, WHITE, [("Raw sources", "public action-labelled photos (V-COCO*)\n48,464 images"),
                                ("Data wrangling", "validity rules CVC1–4 + RT-DETR person/object detector"),
                                ("Benchmark", "152 real + 800 synthetic, persons marked A / B")]),
          ("EXPERIMENT", SKY, NAVY, [("Query generator", "7 caption conditions × Q1–Q4 × 2 answer orders"),
                                     ("VLM inference", "3 open models on an RTX 4050 laptop GPU\n29,928 queries / model"),
                                     ("Logit readout", "P(A) vs P(B), order-averaged; competence set S_m")]),
          ("ANALYSIS", ORANGE, NAVY, [("Statistics", "Wilson CI, McNemar + Holm, trend test, GEE, Beta-Binomial"),
                                      ("Inside the model", "logistic probe, PCA, activation patching"),
                                      ("Outputs", "figures, H1–H5 verdicts, report, live demo")])]
bx0, bw, gap, bh = 1.5, 2.4, 0.35, 1.25
for r_i, (lab, fill, fg, boxes) in enumerate(stages):
    y = 2.15 + r_i * 1.6
    lab_s = rect(s, X0, y, 0.9, bh, fill=fill)
    lab_s.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    bodyPr = lab_s.text_frame._txBody.bodyPr
    bodyPr.set("vert", "vert270")
    fill_tf(lab_s.text_frame, [{"runs": [(lab, dict(size=13, bold=True, color=fg))], "align": PP_ALIGN.CENTER}])
    order = boxes if r_i != 1 else boxes  # row 2 flows right-to-left
    xs = [bx0 + k * (bw + gap) for k in range(3)]
    if r_i == 1:
        xs = xs[::-1]
    for k, ((h, b), x) in enumerate(zip(boxes, xs)):
        card(s, x, y, bw, bh, [[(h, dict(size=14, bold=True, color=fg))], [(b, dict(size=11, color=fg))]],
             fill=fill, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)
    for k in range(2):
        xa, xb = xs[k], xs[k + 1]
        if xa < xb:
            arrow(s, xa + bw + 0.03, y + bh / 2, xb - 0.03, y + bh / 2)
        else:
            arrow(s, xa - 0.03, y + bh / 2, xb + bw + 0.03, y + bh / 2)
# vertical links between rows
xr = bx0 + 2 * (bw + gap) + bw / 2
arrow(s, xr, 2.15 + bh + 0.02, xr, 3.75 - 0.02)
xl = bx0 + bw / 2
arrow(s, xl, 3.75 + bh + 0.02, xl, 5.35 - 0.02)
tb(s, X0, 6.66, W, 0.25, [[("*Pool: 10,346 V-COCO + 38,118 HICO-DET images; 138 of the 152 final real items are V-COCO, 14 HICO-DET.", dict(size=9, color=GREY))]])

# ================================================================ 7. experimental design
s = new_slide("EXPERIMENTAL DESIGN", "Same image, only the text changes: 7 caption conditions × 4 questions",
              notes="Left: the caption ladder. 'None' and 'neutral' are baselines, 'true' states the right person, and "
                    "the four false captions name the bystander with rising confidence, from 'I think B might be' to "
                    "'Verified annotation'. Right: four questions, each asked in a fresh context. Q4 is the binding "
                    "question. Q2 checks the action is still seen (with a foil control), Q3 checks whether the model "
                    "notices the conflict. Every multiple-choice question is asked in both answer orders.")
ladder = [("none", "(image only)", WHITE, INK), ("neutral", "“The image shows A and B.”", RGBColor(0xEE, 0xEE, 0xEE), INK),
          ("true", "“A is throwing the frisbee.”", TINT, NAVY),
          ("hedged", "“I think B might be throwing …”", RGBColor(0xFB, 0xDA, 0xDB), INK),
          ("plain", "“B is throwing the frisbee.”", RGBColor(0xF5, 0xA9, 0xAB), INK),
          ("“definitely”", "“B is definitely throwing …”", RGBColor(0xEC, 0x6A, 0x6E), WHITE),
          ("“verified”", "“Verified annotation: B is … Confirmed: it is B …”", RED, WHITE)]
tb(s, X0, 2.08, 3.7, 0.3, [[("Caption condition (confidence of the lie ↓)", dict(size=12, bold=True, color=NAVY))]])
for k, (lab, txt, fill, fg) in enumerate(ladder):
    y = 2.42 + k * 0.63
    card(s, X0, y, 3.7, 0.56, [[(lab + "  ", dict(size=12, bold=True, color=fg)), (txt, dict(size=11, color=fg, italic=True))]],
         fill=fill, line=LGREY if fill == WHITE else None, anchor=MSO_ANCHOR.MIDDLE, margin=0.08)
picture(s, IMG / "real" / "images" / "vcoco_000000085007_marked.jpg", 4.45, 2.42, h=2.35, border=LGREY)
arrow(s, 4.18, 3.6, 4.42, 3.6, color=NAVY)
tb(s, 4.35, 4.85, 2.5, 1.95, [[("Per image:", dict(size=12, bold=True, color=NAVY))],
                              {"runs": [("fresh context per question", dict(size=12))], "bullet": True},
                              {"runs": [("both orders (A,B) and (B,A)", dict(size=12))], "bullet": True},
                              {"runs": [("≈ 31 queries per image", dict(size=12))], "bullet": True},
                              {"runs": [("29,928 queries per model", dict(size=12, bold=True))], "bullet": True}])
qs = [("Q1 Identity", "“Which marker is on the left person?”", TINT, NAVY),
      ("Q2 Action seen?", "“Is anyone throwing a frisbee?” + foil (absent action)", TINT, NAVY),
      ("Q3 Lie noticed?", "“Does the caption match the image?”", TINT, NAVY),
      ("Q4 Binding", "“Who is throwing the frisbee, A or B?”", NAVY, WHITE)]
for k, (h, b, fill, fg) in enumerate(qs):
    y = 2.42 + k * 1.1
    card(s, 7.0, y, 2.55, 0.98, [[(h, dict(size=13, bold=True, color=fg))], [(b, dict(size=11, color=fg))]], fill=fill,
         anchor=MSO_ANCHOR.MIDDLE, margin=0.08)
arrow(s, 6.62, 3.6, 6.97, 3.6, color=NAVY)

# ================================================================ 8. measurement (equations)
s = new_slide("HOW WE MEASURE", "We read probabilities, not text: four definitions behind every number",
              notes="We never parse free text. (1) From the logits of the first answer token we get the probability of "
                    "'A' versus 'B'. (2) Every question is asked in both answer orders and averaged; this cancels letter "
                    "bias (LLaVA answers 'A' 98% of the time). (3) We only score items the model already gets right "
                    "without any caption: the competence set. (4) The reassignment rate is the share of those items "
                    "where the probability of the true person drops below one half, i.e. the model now names the wrong "
                    "person. The equations are native PowerPoint equations.")
meas = [("1  Logit readout", "probability of ‘A’ from the first answer token’s logits",
         R("P") + D(R("A")) + R("=") + F(SUP(R("e"), SUB(R("ℓ"), R("A"))),
                                          SUP(R("e"), SUB(R("ℓ"), R("A"))) + R("+") + SUP(R("e"), SUB(R("ℓ"), R("B")))),
         "P(A) = e^ℓA / (e^ℓA + e^ℓB)"),
        ("2  Order averaging", "ask (A,B) and (B,A), average: cancels letter bias",
         SUB(R("p"), R("true", False)) + D(R("i") + R(",") + R("c")) + R("=") + F(R("1", False), R("2", False))
         + D(SUP(R("p"), D(R("A,B", False))) + R("+") + SUP(R("p"), D(R("B,A", False))), "[", "]"),
         "p_true(i,c) = ½ [p(A,B) + p(B,A)]"),
        ("3  Competence set", "only items the model gets right with no caption",
         SUB(R("S"), R("m")) + R("=") + D(R("i") + R(" :  ", False) + SUB(R("p"), R("true", False)) + D(R("i") + R(",") + R("none", False))
                                           + R(">") + R("0.5", False) + R(" in both orders, Q2 = yes", False), "{", "}"),
         "S_m = { i : p_true(i, none) > 0.5 in both orders, Q2 = yes }"),
        ("4  Reassignment rate", "share of competent items now given to the wrong person",
         SUB(R("RR", False), R("m")) + D(R("c")) + R("=") + F(R("1", False), D(SUB(R("S"), R("m")), "|", "|"))
         + SUM(R("i") + R("∈") + SUB(R("S"), R("m")),
               R("1", False) + D(SUB(R("p"), R("true", False)) + D(R("i") + R(",") + R("c")) + R("<") + R("0.5", False), "[", "]")),
         "RR_m(c) = (1/|S_m|) Σ 1[p_true(i,c) < 0.5]")]
for k, (h, b, eq, fbk) in enumerate(meas):
    y = 2.1 + k * 1.17
    card(s, X0, y, 2.55, 1.07, [[(h, dict(size=14, bold=True, color=WHITE))], [(b, dict(size=11, color=WHITE))]],
         fill=NAVY if k < 3 else RED, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)
    rect(s, 3.1, y, 6.45, 1.07, fill=TINT if k < 3 else RTINT)
    equation(s, 3.15, y + 0.04, 6.35, 0.99, eq, fbk, size=20 if k != 2 else 17)

# ================================================================ 9. dataset
s = new_slide("DATASET", "Data wrangling: only 0.3% of 48,464 images pass every validity rule",
              notes="Most action datasets are ambiguous for our question: several people, several objects, or both "
                    "people doing the action. We applied the validity conditions in order (log scale). The detector "
                    "steps remove images with an unannotated third person or a second object. 152 real photos survive "
                    "(mostly V-COCO). Real photos give validity; 800 procedurally drawn synthetic scenes give "
                    "statistical power. A/B labels and left/right positions are balanced in both sets.")
keep = ["all images", "has whitelisted action", "CVC1a: exactly 2 persons (annotation)", "CVC1b: separable (IoU<0.3), each >= 2% area",
        "CVC2a: one target object (annotation)", "CVC3: action linked to exactly one person", "CVC4: distractor has no conflicting label",
        "CVC1c: no unannotated person (detector, conf>=0.4)", "CVC2b: one object (annotation U detector, conf>=0.2)"]
short = ["All images", "Has a target action", "Exactly 2 people", "People separable", "One target object",
         "Action on one person", "No conflicting label", "No hidden 3rd person*", "One object only*"]
fv = funnel.set_index("stage").loc[keep, "total"].tolist()
cd = CategoryChartData()
cd.categories = short
cd.add_series("images", fv)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(X0), Inches(2.05), Inches(4.75), Inches(4.55), cd)
ch = gf.chart
style_chart(ch, legend=False)
ch.has_title = False
ch.plots[0].gap_width = 35
ch.plots[0].has_data_labels = True
dl = ch.plots[0].data_labels
dl.number_format, dl.number_format_is_linked, dl.position = "#,##0", False, XL_LABEL_POSITION.OUTSIDE_END
dl.font.size, dl.font.bold, dl.font.color.rgb = Pt(11), True, NAVY
series_color(ch.plots[0].series[0], NAVY)
pt = ch.plots[0].series[0].points[len(fv) - 1]
pt.format.fill.solid()
pt.format.fill.fore_color.rgb = RED
ca = ch.category_axis
ca.reverse_order = True
axis_style(ca, gridlines=False, size=11)
va = ch.value_axis
axis_style(va, fmt="#,##0", size=10, gridlines=False)
va.tick_label_position = XL_TICK_LABEL_POSITION.NONE
va.format.line.fill.background()
scaling = va._element.find(qn("c:scaling"))
lb = etree.Element(qn("c:logBase"))
lb.set("val", "10")
scaling.insert(0, lb)
va.maximum_scale = 400000
va.minimum_scale = 10
tb(s, X0, 6.6, 4.75, 0.3, [[("Log scale. *RT-DETR checks. Pool incl. 38,118 HICO-DET images; †14 of the 152 are HICO-DET.", dict(size=9, color=GREY))]])
p1 = picture(s, IMG / "real" / "images" / "vcoco_000000195809_marked.jpg", 5.4, 2.08, h=1.38, border=LGREY)
p2 = picture(s, IMG / "synthetic" / "images" / "synth_0494_marked.jpg", 5.4 + p1.width / 914400 + 0.12, 2.08, h=1.38, border=LGREY)
tb(s, 5.4, 3.48, 4.15, 0.28, [[("Real: A is typing on the laptop · Synthetic: B is swinging the racket", dict(size=10, color=GREY, italic=True))]])
rows = [["", "Real", "Synthetic"],
        ["Items", "152 photos†", "800 drawn"],
        ["Actions", "20 verbs", "4 × 200"],
        ["Actor is A / left", "55% / 54%", "52% / 52%"],
        ["Max 95% CI", "± 7.9 pp", "± 3.5 pp"]]
table(s, 5.4, 3.82, 4.15, rows, [1.45, 1.35, 1.35], size=12, row_h=0.32, bold_first=True)
card(s, 5.4, 5.55, 4.15, 1.3, [[("Validity rules  ", dict(size=13, bold=True, color=NAVY))],
                               [("CVC1 two separable people · CVC2 one object · CVC3 action on one person · "
                                 "CVC4 no conflicting label · CVC5 3-person audit (R1 done)", dict(size=11))]],
     fill=OTINT, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================ 11. configurations
s = new_slide("CONFIGURATIONS", "Three open models, one frozen protocol, run on a laptop GPU",
              notes="Three open models ran the full experiment on a 6 GB laptop GPU: 29,928 queries each, zero errors. "
                    "Qwen was 4-bit quantised to fit. Prompts and conditions were frozen and pre-registered before "
                    "running, and the statistics were validated on a mock model with known effects.")
rows = [["Model", "Parameters", "Precision", "Readout", "Items", "Queries", "Run time"],
        ["SmolVLM2-2.2B", "2.2 B", "fp16", "logits", "952", "29,928", "≈ 1 h"],
        ["LLaVA-OneVision-0.5B", "0.5 B", "fp16", "logits", "952", "29,928", "3.1 h"],
        ["Qwen2.5-VL-3B", "3 B", "4-bit NF4", "logits", "952", "29,928", "5.0 h"]]
table(s, X0, 2.15, W, rows, [2.3, 1.1, 1.15, 1.0, 0.8, 1.2, 1.55], size=13, row_h=0.45, bold_first=True)
cfg = [("Hardware & software", "RTX 4050 laptop GPU (6 GB) · Python 3.14 · PyTorch 2.11 · transformers 5.23"),
       ("Frozen & pre-registered", "prompts in one config file (hash 973d887011) · fixed seed · images ≤ 640 px · 0 errors in 89,784 queries"),
       ("Quality control", "smoke tests · a mock model with known effects validates the statistics · 3-person audit (R1 done, R2/R3 in progress)")]
for k, (h, b) in enumerate(cfg):
    card(s, X0 + k * 3.08, 4.3, 2.94, 2.2, [[(h, dict(size=15, bold=True, color=NAVY))], [(b, dict(size=13))]],
         fill=[TINT, OTINT, TINT][k])

# ================================================================ 12. demo
s = new_slide("LIVE DEMO", "Change the caption, watch the model's answer move",
              notes="DEMO SCRIPT (2 min). 1) Start before the talk: python src/demo.py, open localhost:8800 (model load "
                    "about 35 s). 2) Click the frisbee or cake image: bars show the probability of the correct person "
                    "under each caption; they turn red as the lie gets more confident. 3) Point at Q2: the model still "
                    "says someone is doing the action. Q3: it says the caption matches. 4) Ask the audience for a "
                    "caption and type it in. If the GPU fails: python src/demo.py --replay shows the recorded answers.")
picture(s, FIGS / "demo.jpg", X0, 2.15, w=5.45, border=LGREY)
steps = [("Pick an image", "real or synthetic, A/B marked"),
         ("Model answers Q4", "under all 7 captions, both orders (≈ 6 s)"),
         ("Bar = p_true", "turns red below 50%: wrong person named"),
         ("Audience types a caption", "Q2/Q3 badges: still sees the action, misses the lie")]
for k, (h, b) in enumerate(steps):
    y = 2.15 + k * 0.98
    chip(s, 6.1, y + 0.1, 0.5, str(k + 1), fill=RED if k == 2 else NAVY, size=15)
    tb(s, 6.7, y, 2.85, 0.9, [[(h, dict(size=14, bold=True, color=NAVY))], [(b, dict(size=12, color=GREY))]], anchor=MSO_ANCHOR.MIDDLE)
card(s, X0, 6.32, 5.45, 0.55, [[("python src/demo.py", dict(size=13, bold=True, font="Courier New", color=WHITE)),
                                ("   backup: --replay (no GPU)", dict(size=12, color=SKY))]], fill=NAVY, anchor=MSO_ANCHOR.MIDDLE)
card(s, 6.1, 6.12, 3.45, 0.75, [[("Live answers match the recorded run to the percent: deterministic and reproducible.",
                                  dict(size=12, color=NAVY, bold=True))]], fill=OTINT, anchor=MSO_ANCHOR.MIDDLE)

# ================================================================ 14. dose response
s = new_slide("RESULTS: DOSE-RESPONSE (H2)", "The more confident the false caption, the more models follow it",
              notes="Main result. x-axis: caption condition; y-axis: reassignment rate, the share of items the model "
                    "got right without text but now gets wrong. Baselines stay near zero; false captions push rates "
                    "up, and the 'verified' wording is the strongest: SmolVLM2 reaches 100%. A GEE logistic "
                    "regression, clustered by image and controlling for model, verb, actor side and person size, "
                    "estimates that each step of confidence multiplies the odds of rebinding by 3.46 on real images "
                    "and 2.22 on synthetic ones.")
for j, ds in enumerate(["real", "synthetic"]):
    cd = CategoryChartData()
    cd.categories = CLAB
    for key, name, _ in MODELS:
        g = rr[(rr.model == key) & (rr.dataset == ds)].set_index("condition")
        cd.add_series(name, [float(g.loc[c, "RR"]) if c in g.index else None for c in CONDS])
    gf = s.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(X0 + j * 4.6), Inches(2.05), Inches(4.5), Inches(3.2), cd)
    ch = gf.chart
    style_chart(ch, legend=False, size=10)
    chart_title(ch, f"{ds.capitalize()} images: reassignment rate", 13)
    for k, (_, _, col) in enumerate(MODELS):
        series_color(ch.plots[0].series[k], col, line=True)
    axis_style(ch.category_axis, gridlines=False, size=9)
    axis_style(ch.value_axis, fmt="0%", maxv=1.0, minv=0.0, size=10)
    ch.value_axis.major_unit = 0.25
for k, (_, name, col) in enumerate(MODELS):
    x = X0 + 1.2 + k * 2.6
    ln = s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x), Inches(5.4), Inches(x + 0.4), Inches(5.4))
    ln.line.color.rgb = col
    ln.line.width = Pt(3)
    tb(s, x + 0.48, 5.26, 1.9, 0.3, [[(name, dict(size=12, color=INK))]], anchor=MSO_ANCHOR.MIDDLE)
rect(s, X0, 5.7, 6.1, 1.17, fill=TINT)
tb(s, X0 + 0.1, 5.72, 3.0, 0.26, [[("GEE logistic regression (clustered by image)", dict(size=10, bold=True, color=NAVY))]])
equation(s, X0 + 0.05, 5.95, 6.0, 0.48,
         R("logit", False) + R(" ", False) + R("P") + D(SUB(R("rebind", False), R("ic", True)))
         + R("=") + SUB(R("β"), R("0", False)) + R("+") + SUB(R("β"), R("1", False)) + R("·") + SUB(R("level", False), R("c"))
         + R("+") + SUB(R("γ"), R("model", False)) + R("+") + SUB(R("δ"), R("verb", False)) + R("+") + R("…", False),
         "logit P(rebind_ic) = β0 + β1·level_c + γ_model + δ_verb + …", size=15)
equation(s, X0 + 0.05, 6.4, 6.0, 0.45,
         R("OR", False) + R("=") + SUP(R("e"), SUB(R("β"), R("1", False))) + R("=") + R("3.46", False)
         + R("  [2.85, 4.18] real,   ", False) + R("2.22", False) + R("  [2.10, 2.34] synthetic", False),
         "OR = e^β1 = 3.46 [2.85, 4.18] real, 2.22 [2.10, 2.34] synthetic", size=14)
card(s, 6.7, 5.7, 2.85, 1.17, [[("100%", dict(size=26, bold=True, color=RED))],
                               [("SmolVLM2 wrong at ‘verified’, real and synthetic", dict(size=11))]],
     fill=RTINT, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)

# ================================================================ 15. H1 table
s = new_slide("RESULTS: PAIRED TEST (H1)", "A plain false caption flips answers far beyond chance",
              notes="The same item with and without the caption gives paired outcomes, so we use McNemar's test: it "
                    "only looks at the discordant pairs, b items that went right-to-wrong and c that went wrong-to-right. "
                    "Holm's correction controls for running many tests; the CI on the difference is bootstrapped. "
                    "H1 holds for SmolVLM2 and Qwen on both datasets. LLaVA only gives in at the strongest wording.")
rows = [["Model", "Data", "n (S_m)", "RR no caption", "RR “B is …”", "Δ [95% CI]", "Holm p", "H1"]]
p = h1[h1.test == "contra_2"]
for key, name, _ in MODELS:
    for ds in ["real", "synthetic"]:
        g = p[(p.model == key) & (p.dataset == ds)].set_index("ref")
        rn = g.loc["none"]
        ok = (g.p_holm < 0.05).all() and (g["diff"] > 0).all()
        pv = rn.p_holm
        ptxt = "< 0.001" if pv < 0.001 else f"{pv:.2f}"
        rows.append([name, ds, f"{int(rn.n)}", f"{rn.RR_ref:.0%}", f"{rn.RR_test:.0%}",
                     f"{rn['diff']*100:+.0f} pp [{rn.diff_lo*100:.0f}, {rn.diff_hi*100:.0f}]", ptxt,
                     ("Yes" if ok else "No", {"bold": True, "color": WHITE, "fill": NAVY if ok else RED, "align": PP_ALIGN.CENTER})])
table(s, X0, 2.1, W, rows, [1.85, 0.95, 0.85, 1.15, 1.15, 1.45, 0.9, 0.8], size=12, row_h=0.38)
rect(s, X0, 4.95, 4.45, 1.9, fill=TINT)
tb(s, X0 + 0.12, 5.0, 4.2, 0.3, [[("McNemar test on paired outcomes", dict(size=13, bold=True, color=NAVY))]])
equation(s, X0 + 0.05, 5.32, 4.35, 0.95,
         SUP(R("χ"), R("2", False)) + R("=") + F(SUP(D(R("b") + R("−") + R("c")), R("2", False)), R("b") + R("+") + R("c")),
         "χ² = (b − c)² / (b + c)", size=22)
tb(s, X0 + 0.12, 6.25, 4.2, 0.55, [[("b = right → wrong after the caption,  c = wrong → right", dict(size=11, color=INK))]])
card(s, 5.05, 4.95, 4.5, 1.9, [{"runs": [("Decision rule", dict(size=13, bold=True, color=NAVY))], "space": 4},
                               {"runs": [("Holm correction across all comparisons", dict(size=12))], "bullet": True},
                               {"runs": [("bootstrap 95% CI on Δ (2,000 resamples)", dict(size=12))], "bullet": True},
                               {"runs": [("H1 needs p < .05 against both ‘none’ and ‘neutral’", dict(size=12))], "bullet": True}],
     fill=OTINT)

# ================================================================ 16. Bayesian
s = new_slide("BAYESIAN ANALYSIS", "The posterior of the reassignment rate moves right as the lie gets more confident",
              notes="Bayesian view from the course: each answer is a Bernoulli trial with unknown rate theta. With a "
                    "uniform Beta(1,1) prior and k wrong answers out of n, the posterior is Beta(1+k, 1+n-k). Each "
                    "curve is one caption level. For all three models, the probability that the 'verified' rate "
                    "exceeds the neutral rate is 1.00.")
picture(s, FIGS / "bayes.png", 1.0, 2.0, w=8.0)
rect(s, X0, 5.85, W, 1.0, fill=OTINT)
equation(s, X0 + 0.05, 5.88, 6.35, 0.94,
         R("θ") + R("∼") + R("Beta", False) + D(R("1,1", False)) + R(",  ", False)
         + R("k") + R("∣") + R("θ") + R("∼") + R("Bin", False) + D(R("n") + R(",") + R("θ"))
         + R("  ⇒  ", False) + R("θ") + R("∣") + R("k") + R("∼") + R("Beta", False)
         + D(R("1", False) + R("+") + R("k") + R(",") + R("1", False) + R("+") + R("n") + R("−") + R("k")),
         "θ ~ Beta(1,1), k|θ ~ Bin(n,θ) ⇒ θ|k ~ Beta(1+k, 1+n−k)", size=14)
equation(s, 6.75, 5.9, 2.8, 0.6,
         R("P") + D(SUB(R("θ"), R("verified", False)) + R(">") + SUB(R("θ"), R("neutral", False))) + R("=") + R("1.00", False),
         "P(θ_verified > θ_neutral) = 1.00", size=13)
tb(s, 6.75, 6.45, 2.8, 0.35, [[("for all three models", dict(size=11, color=GREY))]], name="sub")

# ================================================================ 17. taxonomy H3/H4
s = new_slide("RESULTS: H3 & H4", "Models still see the action; they give it to the wrong person",
              notes="Each bar splits the competent items under the plain false caption. Red: the model names the wrong "
                    "person but still says the action is happening. That is rebinding, not blindness, and it "
                    "dominates for every susceptible model. Orange (action lost) is tiny. H4: SmolVLM2 and LLaVA "
                    "flag the lie in 0-0.4% of cases; only Qwen notices, and noticing partly protects it.")
order = [("smolvlm2-2.2b", "real"), ("smolvlm2-2.2b", "synthetic"), ("qwen2.5-vl-3b", "real"), ("qwen2.5-vl-3b", "synthetic"),
         ("llava-ov-0.5b", "real"), ("llava-ov-0.5b", "synthetic")]
ct = tax.groupby(["model", "dataset"]).outcome.value_counts().unstack(fill_value=0)
outs = [("Stable", "Correct (stable)", LGREY), ("Rebound, perception intact", "Wrong person, action still seen", RED),
        ("Rebound, perception lost", "Wrong person, action lost", ORANGE)]
cd = CategoryChartData()
cd.categories = [f"{MNAME[m].split('-')[0]} · {d[:5]}" for m, d in order]
for o, lab, _ in outs:
    vals = []
    for m, d in order:
        row = ct.loc[(m, d)]
        tot = row.sum() - row.get("Other", 0)
        vals.append(float(row.get(o, 0) / tot) if tot else 0)
    cd.add_series(lab, vals)
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_STACKED_100, Inches(X0), Inches(2.1), Inches(5.65), Inches(4.75), cd)
ch = gf.chart
style_chart(ch, legend=True, size=11)
ch.plots[0].gap_width = 45
ch.plots[0].overlap = 100
for k, (_, _, col) in enumerate(outs):
    series_color(ch.plots[0].series[k], col)
ch.category_axis.reverse_order = True
axis_style(ch.category_axis, gridlines=False, size=11)
axis_style(ch.value_axis, fmt="0%", size=10)
rect(s, 6.3, 2.15, 3.25, 1.6, fill=TINT)
tb(s, 6.4, 2.2, 3.05, 0.3, [[("H3 supported (5/5 cells)", dict(size=14, bold=True, color=NAVY))]])
equation(s, 6.35, 2.52, 3.15, 0.6,
         R("P") + D(R("Q2 = yes", False) + R("∣") + R("rebind", False)) + R("=") + R("0.89–1.00", False),
         "P(Q2 = yes | rebind) = 0.89–1.00", size=14)
tb(s, 6.4, 3.1, 3.05, 0.6, [[("absent actions still rejected 98–100%: not a yes-bias", dict(size=11))]])
pts = [("H4: awareness is rare", "SmolVLM2 and LLaVA flag the lie in 0–0.4% of cases (Q3)", RED, RTINT),
       ("Noticing partly protects", "Qwen flags 62–76% and then rebinds less: odds ratio 0.03 (real), 0.25 (synthetic), Fisher’s exact test", NAVY, OTINT)]
for k, (h, b, col, fill) in enumerate(pts):
    card(s, 6.3, 3.9 + k * 1.5, 3.25, 1.4, [[(h, dict(size=14, bold=True, color=col))], [(b, dict(size=12))]], fill=fill)

# ================================================================ 18. probe / patching
s = new_slide("INSIDE THE MODEL: H5", "In Qwen, the caption overwrites the internal representation itself",
              notes="We cached Qwen's hidden state at every layer for 352 items. A logistic-regression probe trained "
                    "without captions reads the true actor with 90% accuracy from layer 23 on. On the 101 items where "
                    "the false caption flips the answer, the same probe reads the caption's person: the true actor "
                    "only 22% of the time, below chance. So H5 is not supported: the truth is overwritten, not hidden. "
                    "Activation patching shows the route: the caption's 'B' token in early layers, the question tokens "
                    "in layers 18–24, the final token from 27.")
cd = CategoryChartData()
cd.categories = [str(int(l)) for l in probe.layer]
cd.add_series("No caption", probe.acc_none.round(3).tolist())
cd.add_series("False caption, answer flipped", probe.acc_contra3_flipped.round(3).tolist())
cd.add_series("Chance", [0.5] * len(probe))
gf = s.shapes.add_chart(XL_CHART_TYPE.LINE, Inches(X0), Inches(2.08), Inches(4.65), Inches(3.65), cd)
ch = gf.chart
style_chart(ch, legend=True, size=10)
chart_title(ch, "Logistic probe: accuracy for the true actor", 12)
series_color(ch.plots[0].series[0], BLUE, line=True)
series_color(ch.plots[0].series[1], RED, line=True)
series_color(ch.plots[0].series[2], RGBColor(0x99, 0x99, 0x99), line=True, width=1.25)
from pptx.enum.dml import MSO_LINE_DASH_STYLE
ch.plots[0].series[2].format.line.dash_style = MSO_LINE_DASH_STYLE.DASH
for sr in ch.plots[0].series:
    sr.marker.style = None
    from pptx.enum.chart import XL_MARKER_STYLE
    sr.marker.style = XL_MARKER_STYLE.NONE
ca = ch.category_axis
axis_style(ca, gridlines=False, size=9)
ca.tick_labels.font.size = Pt(9)
ca_el = ca._element
for tag, val in (("c:tickLblSkip", "5"), ("c:tickMarkSkip", "5")):
    e = ca_el.find(qn(tag))
    if e is None:
        e = etree.SubElement(ca_el, qn(tag))
    e.set("val", val)
axis_style(ch.value_axis, fmt="0%", maxv=1.0, minv=0.0, size=10)
tb(s, X0, 5.68, 4.65, 0.25, [[("x-axis: decoder layer of Qwen2.5-VL-3B (36 layers)", dict(size=10, color=GREY))]])
picture(s, FIGS / "patching.png", 5.2, 2.08, w=4.35)
tb(s, 5.2, 4.35, 4.35, 1.4, [{"runs": [("Patching route: ", dict(size=12, bold=True, color=NAVY)),
                                       ("caption’s “B” (layers 0–15) → question tokens (18–24) → final token (27+).", dict(size=12))], "space": 4},
                             {"runs": [("Knockout: ", dict(size=12, bold=True, color=NAVY)),
                                       ("blocking direct final→“B” attention changes nothing.", dict(size=12))], "space": 4},
                             {"runs": [("PCA: ", dict(size=12, bold=True, color=NAVY)),
                                       ("57 of 2,048 dimensions hold 90% of the variance; PC1 (28%) is image style, not the actor.", dict(size=12))]}])
res3 = [("90%", "probe accuracy from layer 23, no caption", BLUE, TINT),
        ("22%", "true actor on flipped items [15–31%]: probe reads the caption’s person", RED, RTINT),
        ("H5 ✗", "not supported: the truth is overwritten, not hidden", NAVY, OTINT)]
for k, (big, small, col, fill) in enumerate(res3):
    card(s, X0 + k * 3.07, 5.98, 2.95, 0.88, [[(big + "  ", dict(size=20, bold=True, color=col)), (small, dict(size=11))]],
         fill=fill, anchor=MSO_ANCHOR.MIDDLE, margin=0.1)

# ================================================================ 21. analysis
s = new_slide("ANALYSIS & KEY OBSERVATIONS", "Three hypotheses supported, one partly, one rejected",
              notes="Summary of the five pre-registered hypotheses, then three observations. Real photos are harder "
                    "than synthetic ones, size alone does not predict safety, and a simple instruction is not a fix.")
SUP, PART, NOT = (NAVY, "Supported"), (DORANGE, "Partly"), (RED, "Not supported")
rows = [["H", "Claim", "Test", "Verdict", "Evidence"],
        [("H1", {"bold": True}), "A false caption flips ‘who’", "McNemar + Holm", SUP,
         "SmolVLM2 & Qwen on both datasets; LLaVA only at ‘verified’"],
        [("H2", {"bold": True}), "Effect grows with confidence", "Trend test, GEE", SUP,
         "odds ×3.46 (real), ×2.22 (synth.) per step"],
        [("H3", {"bold": True}), "Rebinding, not blindness", "P(Q2 | rebind), foil", SUP,
         "89–100% still see the action"],
        [("H4", {"bold": True}), "Noticing does not protect", "Odds ratio, Fisher", PART,
         "small models never notice; Qwen partly protected"],
        [("H5", {"bold": True}), "The truth survives inside", "Probe, patching", NOT,
         "probe reads the caption’s person (22%)"]]
rows = [r if i == 0 else [r[0], r[1], r[2], (r[3][1], {"bold": True, "color": WHITE, "fill": r[3][0], "align": PP_ALIGN.CENTER}), r[4]]
        for i, r in enumerate(rows)]
table(s, X0, 2.1, W, rows, [0.5, 2.3, 1.75, 1.3, 3.25], size=12, row_h=0.5)
obs = [("Real photos are harder", "Qwen rebinds 79% on real vs 27% on synthetic under the plain lie", RTINT, RED),
       ("Size ≠ safety", "LLaVA-0.5B, the smallest, resists until ‘verified’, then follows 96% (real)", TINT, NAVY),
       ("Prompting is no fix", "“Answer from the image” helps SmolVLM2 8 pp, makes Qwen 7 pp worse", OTINT, DORANGE)]
for k, (h, b, fill, col) in enumerate(obs):
    card(s, X0 + k * 3.07, 5.3, 2.95, 1.55, [[(h, dict(size=14, bold=True, color=col))], [(b, dict(size=12))]], fill=fill)

# ================================================================ 24. conclusion
s = new_slide("CONCLUSION", None,
              notes="Close with the one-sentence message, then the three takeaways. Invite questions.")
card(s, X0, 1.75, W, 2.2, [[("Small vision-language models let one confident sentence override what they see.", dict(size=22, bold=True, color=WHITE))],
                          {"runs": [("They still perceive the action but bind it to the person the caption names, and inside the model the "
                                     "truth is overwritten, not hidden.", dict(size=16, color=WHITE))], "before": 8}],
     fill=NAVY, anchor=MSO_ANCHOR.MIDDLE, margin=0.3)
tk = [("Data", "48,464 raw images wrangled into 952 controlled items", ORANGE),
      ("Evidence", "89,784 queries, 5 pre-registered hypotheses; paired, regression and Bayesian statistics", SKY),
      ("Implication", "don't trust a small VLM's answer when its prompt contains unverified text", RED)]
for k, (h, b, col) in enumerate(tk):
    x = X0 + k * 3.07
    rect(s, x, 4.25, 2.95, 0.12, fill=col, shape=MSO_SHAPE.RECTANGLE)
    tb(s, x, 4.45, 2.95, 1.4, [[(h, dict(size=18, bold=True, color=NAVY))], [(b, dict(size=14))]])
tb(s, X0, 5.85, W, 0.35, [{"runs": [("Limits: 152 real items (target 400) · models ≤ 3 B · audit 1 of 3 annotators done · "
                                     "next: 7 B models, human baseline", dict(size=11, color=GREY))], "align": PP_ALIGN.CENTER}])

tb(s, X0, 6.25, W, 0.55, [{"runs": [("Thank you · Questions?", dict(size=22, bold=True, color=TITLE))], "align": PP_ALIGN.CENTER}],
   anchor=MSO_ANCHOR.MIDDLE)

# ================================================================ 25. references
s = new_slide("REFERENCES", None, notes="Datasets, models, methods and the course textbooks.")
refs = [
    "Chao, Y.-W., Liu, Y., Liu, X., Zeng, H., Deng, J. (2018). Learning to detect human–object interactions. WACV. [HICO-DET]",
    "Gupta, S., Malik, J. (2015). Visual semantic role labeling. arXiv:1505.04474. [V-COCO]",
    "Lin, T.-Y. et al. (2014). Microsoft COCO: Common objects in context. ECCV.",
    "Li, Y.-L. et al. (2020). PaStaNet: Toward human activity knowledge engine. CVPR. [HAKE]",
    "Zhao, Y. et al. (2024). DETRs beat YOLOs on real-time object detection. CVPR. [RT-DETR]",
    "Marafioti, A. et al. (2025). SmolVLM: Redefining small and efficient multimodal models. arXiv:2504.05299.",
    "Li, B. et al. (2024). LLaVA-OneVision: Easy visual task transfer. arXiv:2408.03326.",
    "Bai, S. et al. (2025). Qwen2.5-VL technical report. arXiv:2502.13923.",
    "Yang, J. et al. (2023). Set-of-Mark prompting unleashes visual grounding in GPT-4V. arXiv:2310.11441.",
    "Alain, G., Bengio, Y. (2016). Understanding intermediate layers using linear classifier probes. arXiv:1610.01644.",
    "Meng, K., Bau, D., Andonian, A., Belinkov, Y. (2022). Locating and editing factual associations in GPT. NeurIPS.",
    "McNemar (1947) Psychometrika; Wilson (1927) JASA; Fleiss (1971) Psych. Bulletin; Holm (1979) Scand. J. Stat.; Liang & Zeger (1986) Biometrika.",
    "Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer. [RB1]",
    "Blum, A., Hopcroft, J., Kannan, R. (2020). Foundations of Data Science. Cambridge University Press. [TB1]",
]
tb(s, X0, 1.62, W, 5.3, [{"runs": [(f"[{i+1}]  ", dict(size=12, bold=True, color=NAVY)), (r, dict(size=12))], "space": 4}
                          for i, r in enumerate(refs)])

prs.save(str(OUT))
print("saved", OUT, len(prs.slides), "slides")
