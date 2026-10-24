"""Fill the supplied course report template with the verified project-three results.

The source document is retained. The builder creates a separate final DOCX and
keeps the existing layout, styles, tables, footer and project-one/two evidence.
"""
from __future__ import annotations

import copy
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from docx.text.paragraph import Paragraph
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "docs" / "reports"
SOURCE = REPORT_DIR / "source" / "行业工程实践_项目一、二已完成.docx"
OUTPUT = REPORT_DIR / "current" / "行业工程实践_项目一至三已完成.docx"
ARCH_IMAGE = ROOT / "outputs" / "stage22" / "stage22_project3_architecture.png"
VISION_RESULT = ROOT / "outputs" / "stage22" / "stage22_vision_result.png"
VISION_MOBILE = ROOT / "outputs" / "stage22" / "stage22_vision_mobile.png"
FONT_REGULAR = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"


def ptext(paragraph: Paragraph) -> str:
    return paragraph.text.strip()


def find_para(
    doc: Document,
    prefix: str,
    start: int = 0,
    *,
    skip_toc: bool = False,
) -> Paragraph:
    for paragraph in doc.paragraphs[start:]:
        if skip_toc and paragraph.style.name.startswith("toc"):
            continue
        if ptext(paragraph).startswith(prefix):
            return paragraph
    raise ValueError(f"Paragraph not found: {prefix}")


def find_table(doc: Document, first_header: str):
    for table in doc.tables:
        if table.rows and table.rows[0].cells[0].text.strip() == first_header:
            return table
    raise ValueError(f"Table not found: {first_header}")


def clear_paragraph_content(paragraph: Paragraph) -> None:
    for child in list(paragraph._p):
        if child.tag != qn("w:pPr"):
            paragraph._p.remove(child)


def copy_paragraph(template: Paragraph, text: str) -> OxmlElement:
    element = copy.deepcopy(template._p)
    paragraph = Paragraph(element, template._parent)
    clear_paragraph_content(paragraph)
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(12)
    return element


def insert_paragraphs_before(
    anchor: Paragraph,
    texts: list[str],
    template: Paragraph,
) -> None:
    for text in texts:
        anchor._p.addprevious(copy_paragraph(template, text))


def replace_paragraph(paragraph: Paragraph, text: str) -> None:
    clear_paragraph_content(paragraph)
    run = paragraph.add_run(text)
    run.font.name = "Times New Roman"
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
    run.font.size = Pt(12)


def replace_prompt_with_paragraphs(
    doc: Document,
    prefix: str,
    texts: list[str],
    body_template: Paragraph,
) -> None:
    prompt = find_para(doc, prefix)
    insert_paragraphs_before(prompt, texts, body_template)
    prompt._p.getparent().remove(prompt._p)


def set_cell(cell, text: str) -> None:
    paragraph = cell.paragraphs[0]
    rpr = None
    for run in paragraph.runs:
        if run._element.rPr is not None:
            rpr = copy.deepcopy(run._element.rPr)
            break
    clear_paragraph_content(paragraph)
    run = paragraph.add_run(text)
    if rpr is not None:
        run._element.insert(0, rpr)
    else:
        run.font.name = "Times New Roman"
        run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
        run.font.size = Pt(9)


def fill_row(table, row_index: int, values: list[str]) -> None:
    cells = table.rows[row_index].cells
    if len(cells) != len(values):
        raise ValueError(
            f"Row {row_index} has {len(cells)} cells, got {len(values)} values")
    for cell, value in zip(cells, values):
        set_cell(cell, value)


def set_table_widths(table, widths: list[float]) -> None:
    """Set fixed, intentional column widths within the 5.77-inch text width."""
    if len(widths) != len(table.columns):
        raise ValueError(
            f"Table has {len(table.columns)} columns, got {len(widths)} widths")
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Inches(width)


def keep_table_rows_together(table) -> None:
    for row in table.rows:
        tr_pr = row._tr.get_or_add_trPr()
        if tr_pr.find(qn("w:cantSplit")) is None:
            tr_pr.append(OxmlElement("w:cantSplit"))
    if len(table.rows) <= 5:
        for row in table.rows[:-1]:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.keep_with_next = True


def insert_picture_before(
    doc: Document,
    anchor: Paragraph,
    image_path: Path,
    width_inches: float,
    caption: str | None,
    body_template: Paragraph,
    caption_template: Paragraph,
) -> None:
    image_el = copy.deepcopy(body_template._p)
    anchor._p.addprevious(image_el)
    image_para = Paragraph(image_el, body_template._parent)
    clear_paragraph_content(image_para)
    image_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_para.paragraph_format.keep_with_next = True
    image_para.paragraph_format.line_spacing = 1.0
    image_para.paragraph_format.space_before = Pt(6)
    image_para.paragraph_format.space_after = Pt(4)
    image_para.add_run().add_picture(str(image_path), width=Inches(width_inches))
    if caption:
        anchor._p.addprevious(copy_paragraph(caption_template, caption))


def insert_code_before(
    anchor: Paragraph,
    lines: list[str],
    template: Paragraph,
) -> None:
    for line in lines:
        element = copy_paragraph(template, line)
        paragraph = Paragraph(element, template._parent)
        paragraph.paragraph_format.first_line_indent = None
        paragraph.paragraph_format.line_spacing = 1.0
        paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for run in paragraph.runs:
            run.font.name = "Courier New"
            run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
            run.font.size = Pt(9)
        anchor._p.addprevious(element)


def build_architecture_image(path: Path) -> None:
    """Create a print-friendly diagram matching the implemented bridge topology."""
    width, height = 1200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(FONT_REGULAR, 30)
    box_font = ImageFont.truetype(FONT_REGULAR, 25)
    small_font = ImageFont.truetype(FONT_REGULAR, 21)
    tiny_font = ImageFont.truetype(FONT_REGULAR, 17)
    micro_font = ImageFont.truetype(FONT_REGULAR, 15)
    fill = "#F4F4F4"
    stroke = "#303030"

    def box(x1, y1, x2, y2, title, detail="", *, font=box_font,
            detail_font=None):
        draw.rounded_rectangle((x1, y1, x2, y2), radius=8, fill=fill,
                               outline=stroke, width=2)
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text(((x1 + x2 - (bbox[2] - bbox[0])) / 2, y1 + 10),
                  title, font=font, fill="#111111")
        if detail:
            dfont = detail_font or tiny_font
            bbox = draw.textbbox((0, 0), detail, font=dfont)
            draw.text(
                ((x1 + x2 - (bbox[2] - bbox[0])) / 2, y1 + 38),
                detail, font=dfont, fill="#111111")

    def arrow(x1, y1, x2, y2):
        draw.line((x1, y1, x2, y2), fill=stroke, width=3)
        if x1 == x2:
            direction = -1 if y2 < y1 else 1
            draw.polygon(
                [(x2, y2), (x2 - 8, y2 - 14 * direction),
                 (x2 + 8, y2 - 14 * direction)],
                fill=stroke,
            )
        else:
            direction = -1 if x2 < x1 else 1
            draw.polygon(
                [(x2, y2), (x2 - 14 * direction, y2 - 8),
                 (x2 - 14 * direction, y2 + 8)],
                fill=stroke,
            )

    draw.text((34, 16), "多模态个性化推荐助手总体架构", font=title_font, fill="black")
    inputs = [
        ("文字输入", "品类 / 预算 / 关键词"),
        ("语音输入", "SenseVoiceSmall → 文本"),
        ("图片输入", "拍照 / 上传 → JPEG"),
        ("手势反馈", "MediaPipe → 喜欢 / 不喜欢"),
    ]
    for index, (title, detail) in enumerate(inputs):
        x1 = 30 + index * 292
        box(x1, 62, x1 + 264, 142, title, detail,
            font=small_font, detail_font=tiny_font)
        arrow(x1 + 132, 142, x1 + 132, 163)

    draw.rounded_rectangle((22, 163, 1178, 280), radius=9, fill="#FAFAFA",
                           outline=stroke, width=3)
    draw.text((38, 174), "项目一 Flask 页面与接口层  ·  127.0.0.1:5000",
              font=box_font, fill="black")
    apis = [
        ("/api/search", "解析 → Top-K"),
        ("/api/voice", "WAV → 文本"),
        ("/api/gesture", "JPEG → 手势"),
        ("/api/vision/analyze", "图片 → 视觉线索"),
        ("/feedback · /api/tts", "反馈 · 播报"),
    ]
    for index, (title, detail) in enumerate(apis):
        x1 = 38 + index * 226
        box(x1, 207, x1 + 206, 264, title, detail,
            font=tiny_font, detail_font=micro_font)

    draw.rounded_rectangle((22, 318, 705, 610), radius=9, fill="#F8F8F8",
                           outline=stroke, width=3)
    draw.text((38, 329), "项目一 · SmartSearch 推荐链路",
              font=title_font, fill="black")
    stages = [
        ("① 查询约束", "类目、价格、关键词\n视觉结果先预览确认"),
        ("② 候选过滤", "硬约束过滤\n剔除点踩商品"),
        ("③ 个性化排序", "0.50 相似度 + 0.25 热度\n+ 0.25 反馈"),
        ("④ 结果解释", "真实字段生成理由\n输出 Top-K"),
    ]
    for index, (title, detail) in enumerate(stages):
        x1 = 38 + (index % 2) * 325
        y1 = 395 + (index // 2) * 100
        box(x1, y1, x1 + 292, y1 + 76, title, "", font=tiny_font)
        draw.multiline_text((x1 + 18, y1 + 39), detail, font=micro_font,
                            fill="#222222", spacing=7)

    draw.rounded_rectangle((728, 318, 1178, 610), radius=9, fill="#F8F8F8",
                           outline=stroke, width=3)
    draw.text((744, 329), "项目二 · 本地多模态桥接  :8100",
              font=title_font, fill="black")
    services = [
        ("/asr", "SenseVoiceSmall"),
        ("/gesture", "MediaPipe Hands"),
        ("/vision", "Qwen2.5-VL-3B"),
        ("/llm", "Ollama qwen2.5:3b"),
        ("/tts", "edge-tts"),
    ]
    for index, (api, model) in enumerate(services):
        row, col = index // 2, index % 2
        x1 = 746 + col * 210
        y1 = 392 + row * 67
        box(x1, y1, x1 + 196, y1 + 54, api, model,
            font=tiny_font, detail_font=micro_font)

    arrow(705, 465, 728, 465)
    draw.rounded_rectangle((22, 640, 1178, 704), radius=9, fill="#FAFAFA",
                           outline=stroke, width=3)
    draw.text((38, 649), "共享状态与工程边界", font=box_font, fill="black")
    bottoms = [
        ("本地资产", "画像 · 索引 · 用户向量 · 模型"),
        ("安全与隐私", "密钥不入报告 · 原图不落盘"),
        ("降级与验收", "桥接不可用返回 503"),
    ]
    bottom_font = ImageFont.truetype(FONT_REGULAR, 17)
    for index, (title, detail) in enumerate(bottoms):
        x1 = 38 + index * 380
        draw.rounded_rectangle(
            (x1, 648, x1 + 350, 697), radius=7, fill=fill,
            outline=stroke, width=2)
        draw.text((x1 + 12, 654), title, font=small_font, fill="#111111")
        draw.text((x1 + 12, 677), detail, font=bottom_font, fill="#222222")

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=True)


def update_update_fields(doc: Document) -> None:
    settings = doc.settings.element
    update = settings.find(qn("w:updateFields"))
    if update is None:
        update = OxmlElement("w:updateFields")
        settings.append(update)
    update.set(qn("w:val"), "true")


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)
    if not VISION_RESULT.exists() or not VISION_MOBILE.exists():
        raise FileNotFoundError("Missing vision validation screenshots")

    build_architecture_image(ARCH_IMAGE)
    doc = Document(SOURCE)
    body_template = find_para(doc, "本项目面向零售电商交易数据")
    caption_template = find_para(doc, "图3-1")

    # Cover date and template-only guidance.
    cover_date = find_para(doc, "__2026__年__9__月")
    replace_paragraph(cover_date, "__2026__年__10__月")
    for prefix in ("【模板使用说明】", "正文建议统一采用宋体小四"):
        paragraph = find_para(doc, prefix)
        paragraph._p.getparent().remove(paragraph._p)

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：结合智能科学与技术",
        [
            "本实训面向智能科学与技术（专升本）专业，以电商商品推荐与多模态交互为业务场景，"
            "要求学生在四周、96学时内完成数据处理、推荐算法、服务封装、多模态交互和端到端验收。"
            "课程并不只考察某个模型能否运行，更要求说明需求如何转化为接口、指标如何复现、"
            "异常如何降级，以及每个结论能否从代码、日志或测试记录中找到依据。",
            "最终系统由项目一的商品推荐服务和项目二的智能多模态助手组成。项目一形成"
            "“数据清洗—ItemCF/Item2Vec召回—LR/LightGBM精排—Flask服务—反馈闭环”链路；"
            "项目二形成“文本/语音/图像输入—VAD与ASR—LLM/RAG—视觉理解—TTS—界面编排”能力。"
            "项目三通过本地桥接服务把两端连接起来，用户可以用文字、语音或一张图片表达需求，"
            "查看可编辑的推荐条件，再以点赞、点踩或手势反馈参与下一轮排序。",
        ],
        body_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明目标用户",
        [
            "目标用户是需要在较短时间内找到具体商品的普通消费者，典型环境包括学生宿舍、"
            "家庭和移动端浏览。第一类场景是用户看到需要添置的物品，但难以准确描述类别，"
            "系统通过拍照识别冰箱、洗衣机等可见物体，再结合“200元以内”等预算完成推荐。"
            "第二类场景是用户直接说出或输入需求，系统识别品类、价格和关键词，返回带真实理由的Top-K商品。"
            "第三类场景是用户浏览结果后用点赞、点踩或手势表达偏好，系统保存反馈并调整下一轮排序。",
            "系统边界限定在本地演示环境：输入支持文字、语音、图片和手势；推荐使用现有商品库；"
            "视觉模型只提取可见场景线索，不执行人物识别，也不推断品牌、材质和价格；"
            "TTS依赖edge-tts网络服务，知识库RAG只在项目二助手内使用，不替代项目一的推荐证据。",
        ],
        body_template,
    )
    table = find_table(doc, "需求编号")
    fill_row(table, 1, [
        "R1", "功能需求",
        "支持文字/语音提出条件；支持图片提取场景、物体和合法类目；用户确认后返回个性化Top-K；"
        "支持点赞、点踩和手势反馈。",
        "页面演示、接口测试、端到端用例",
    ])
    fill_row(table, 2, [
        "R2", "非功能需求",
        "视觉或桥接服务不可用时不阻塞文字与语音推荐；错误返回明确状态码和提示；"
        "关键指标可由固定种子与outputs产物复现。",
        "响应时间记录、降级用例、健康检查",
    ])
    fill_row(table, 3, [
        "R3", "工程约束",
        "使用本地模型与本地HTTP桥接；原始图片不落盘；报告不包含密钥；"
        "项目一、项目二保持独立conda环境与原有接口语义。",
        "代码检查、隐私检查、运行日志",
    ])

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：概述三个",
        [
            "三个32学时模块按依赖关系推进。项目一先解决“推荐什么”，交付可查询、可排序、"
            "可反馈的商品推荐服务；项目二解决“如何自然交互”，交付语音、文本、图像、手势与播报能力；"
            "项目三解决“如何把两端安全地接起来”，新增视觉选品接口和本地桥接端点，"
            "但保持原有推荐接口与排序逻辑不变。",
            "项目三达到的主要结果包括：视觉模型从家电示意图中识别出“家用电器”等合法类目；"
            "保留用户“200元以内”的条件后返回4条带理由的推荐；浏览器端完成拍照、图片上传、"
            "结果预览、人工修改和确认搜索；桥接服务关闭时视觉接口明确返回503，文字与语音链路仍可使用。",
        ],
        body_template,
    )
    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：每组需明确分工",
        [
            "本人邱子健承担推荐链路、项目二核心模块、项目三桥接与网页集成的主要开发工作，"
            "并整理阶段日志、测试记录和报告证据。秦兴睿参与ModalRouter手势动作映射、"
            "环境检查项补充和文字校对。表1-3按实际交付记录分工，未把组内其他成员的工作写入个人成果。",
        ],
        body_template,
    )

    # Project two handoff now records the implemented project three state.
    handoff = find_para(doc, "与项目三的衔接：")
    replace_paragraph(
        handoff,
        "与项目三的衔接：项目三已经通过本地桥接服务使用项目二的SenseVoiceSmall、"
        "MediaPipe、Qwen2.5-VL、Ollama和edge-tts能力。语音结果直接进入现有查询框，"
        "视觉结果先转换为可见物体、颜色和真实商品类目，再由用户确认；"
        "手势仍通过同一反馈接口写入数据库。项目二无需直接导入项目一代码，两边以127.0.0.1:8100"
        "的HTTP协议解耦，既避免顶层config包冲突，也保留了各自环境的独立性。",
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：确定2～3个典型",
        [
            "场景一是宿舍生活用品补购。用户在智能推荐页拍下冰箱、洗衣机或桌面环境，"
            "系统识别可见物体并映射到“家用电器”“宿舍用品”等类目。用户可补充“200元以内”，"
            "确认“200元以内 家用电器 冰箱 洗衣机 扫地机器人”后进入现有推荐链路。",
            "场景二是预算明确的即时查询。用户用文字或语音说出“不超过3元的厨房用品”或"
            "“5000元以上的苹果手机”，规则解析优先提取品类、价格和关键词；规则无法覆盖时，"
            "再调用本地qwen2.5:3b补充约束，桥接失败则回退纯规则。",
            "场景三是结果反馈。用户对某商品点赞、点踩，或对着摄像头竖起/放下大拇指，"
            "反馈写入SQLite后影响下一轮相似度、热度与反馈加权结果。系统边界是不识别个人身份，"
            "不保存原始图片，不承诺离线TTS，也不把视觉模型的描述当成商品事实。",
            "验收指标包括：视觉响应返回scene、objects、colors、category_hints和search_query；"
            "类目必须属于真实商品库；用户价格条件不得被视觉结果覆盖；正常样例返回非空Top-K；"
            "缺少图片返回400，超过8MB返回413，桥接或模型不可用返回503。"
        ],
        body_template,
    )

    figure_caption = find_para(doc, "图5-1")
    old_picture = figure_caption._p.getprevious()
    if old_picture is not None and old_picture.xpath(".//a:blip"):
        old_picture.getparent().remove(old_picture)
    insert_paragraphs_before(
        figure_caption,
        [
            "系统采用两个本地进程。项目一Flask服务运行在127.0.0.1:5000，负责页面、登录、"
            "查询解析、推荐排序和反馈；项目二桥接服务运行在127.0.0.1:8100，负责ASR、"
            "MediaPipe手势、Qwen2.5-VL视觉、Ollama文本模型和TTS。项目一只发送HTTP请求，"
            "不直接导入项目二代码，因此两个conda环境中的同名config包不会冲突。",
            "视觉主链为：浏览器拍照或选图，最长边缩至1024像素并上传；项目一校验登录、"
            "8MB上限和JPEG/PNG格式，把真实商品类目列表发给桥接服务；桥接端再次把图片缩到"
            "最长边768像素、重编码为JPEG以去除EXIF，再调用qwen2.5vl:3b。模型输出先经过JSON提取、"
            "字段收敛、类目交集和长度限制，页面展示后再由用户确认搜索。",
        ],
        body_template,
    )
    insert_picture_before(
        doc, figure_caption, ARCH_IMAGE, 5.60,
        None,
        body_template, caption_template,
    )

    interface_table = find_table(doc, "接口/模块")
    fill_row(interface_table, 1, [
        "多模态助手→推荐服务",
        "multipart：image、可选query；图片≤8MB",
        "scene、objects、colors、category_hints、suggested_query",
        "浏览器→Flask→桥接/vision",
        "格式错误400；过大413；桥接或模型不可用503，保留文字搜索",
    ])
    fill_row(interface_table, 2, [
        "推荐服务→助手",
        "确认后的q、用户会话、top_n",
        "constraints、Top-K item_id/name/category/price/score/reason",
        "Flask SmartSearch与结果模板",
        "无结果返回提示；点踩商品剔除；原图不参与排序",
    ])
    fill_row(interface_table, 3, [
        "知识库/解释模块",
        "视觉可见字段；用户历史与反馈；推荐得分",
        "场景证据、偏好匹配、来源与reason_detail",
        "辅助生成可核验说明",
        "信息不足时明确说明，不补写价格、品牌或属性",
    ])
    fill_row(interface_table, 4, [
        "反馈模块",
        "username、item_id、action(like/dislike)",
        "state、stats、hint",
        "浏览器按钮或手势→/feedback",
        "重复同动作取消；点踩剔除；未登录401",
    ])

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明系统如何从文本",
        [
            "文字查询先由query_parser执行规则解析。正则表达式优先识别价格区间、上下限和单位，"
            "类目别名映射到商品库真实类目，剩余文本清洗请求词后作为商品名或编码关键词。"
            "规则已提取明确条件时不调用大模型，以降低延迟；候选太少、只剩主观词或没有有效约束时，"
            "再通过项目二桥接调用本地qwen2.5:3b，最后按“价格整对取规则、品类和关键词取并集”的规则合并。",
            "语音输入先由SenseVoiceSmall转写成UTF-8文本，语音本身不携带额外推荐参数，"
            "而是复用同一条文字解析路径。图片输入由视觉模型输出可见物体、颜色、合法类目和建议搜索词；"
            "视觉选品只使用可核验的物品与类目，预算由用户在可编辑搜索框单独输入，"
            "不把模型猜测的价格当成商品事实。查询默认只加入首个主类目和最多三个物体名。"
            "颜色作为核验证据展示，不默认进入查询，因为现有关键词采用任一命中，高频颜色会挤掉物体线索。",
            "参数缺失时按无该条件处理并回退热门或偏好；图片无法解码返回400；"
            "用户文字条件与视觉类目冲突时不做静默覆盖，保留用户条件和视觉类目供人工修改。"
            "所有视觉请求固定等待上限180秒，页面可取消，超时后不提交搜索。"
        ],
        body_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：根据用户偏好",
        [
            "项目三没有新增复杂深度排序模型，而是复用项目一的可解释流程。硬约束阶段按类目、"
            "价格上下限和关键词过滤商品；SessionContext合并本轮或追问条件，"
            "Orchestrator再调用SmartSearch。反馈阶段先剔除当前会话点踩商品，再把基础客户向量、"
            "0.8×点赞均值与0.35×点踩惩罚合成为规范化偏好向量。候选得分保持"
            "“0.50×相似度 + 0.25×热度 + 0.25×反馈”，并为每条结果保留来源和EvidencePack。",
            "调整反馈前，查询“不超过3元的厨房用品”的首条商品得分为0.497；点赞后反馈项变为1.0，"
            "该商品得分升至0.906，理由从“符合你的偏好”变为“你点赞过”，相似商品也获得额外反馈分。"
            "项目三继续使用这条已验证规则，使文字、语音和手势反馈落到同一数据模型，"
            "避免不同入口产生不同排序语义。",
        ],
        body_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明多模态助手如何调用",
        [
            "项目一Flask在/api/vision/analyze中读取multipart图片和可选文字条件，校验后调用"
            "MultimodalClient.vision；客户端把JPEG/PNG编码为base64，并携带SmartSearch的真实类目列表"
            "请求项目二/vision。桥接返回结构化JSON，项目一再次收敛字段并生成suggested_query。"
            "页面先展示图片、场景、物体、颜色和类目，用户编辑文本框后才跳转/search。",
            "推荐理由只使用真实字段：命中类目、价格约束、商品画像中的均价与购买人数、"
            "用户向量相似度、点赞相似度或首页来源商品。视觉识别的“冰箱”“洗衣机”可以解释为"
            "来自图片的可见物体，但不能据此编造商品品牌或材质。若视觉结果为空，页面提示修改条件，"
            "不生成虚构识别结果。",
        ],
        body_template,
    )

    feedback_caption_anchor = find_para(
        doc, "5.6、用户反馈与推荐结果对照", skip_toc=True)
    insert_picture_before(
        doc, feedback_caption_anchor, VISION_RESULT, 5.60,
        "图5-2  视觉图片识别结果与可编辑推荐条件",
        body_template, caption_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明点赞、偏好选择",
        [
            "点赞、点踩和手势反馈都调用同一个/feedback接口。点赞写入用户级SQLite并可长期复用；"
            "点踩与继续浏览只进入当前session_id，不能污染其他会话。重复同一动作取消，点踩会覆盖点赞"
            "且立即从当前会话下一轮候选中剔除。"
            "为比较反馈前后结果，使用同一用户qzj、同一请求“200元以内 家用电器 冰箱 洗衣机"
            " 扫地机器人”和相同候选条件，仅改变会话反馈状态。",
            "反馈前4件候选商品的得分依次为0.4853、0.4509、0.4277、0.4124；点赞首件商品后，"
            "四件商品得分依次升至0.8487、0.8056、0.7803、0.7574。首件理由变为“你点赞过”，"
            "其余相似商品反馈项达到0.97以上。变化方向符合预期，但当前样例候选只有4件，"
            "K=5时顺序变化而没有增加新商品，不能据此宣称覆盖更多品类。",
        ],
        body_template,
    )
    comparison_table = find_table(doc, "对照项")
    fill_row(comparison_table, 1, [
        "反馈前",
        "qzj；200元以内 家用电器 冰箱 洗衣机 扫地机器人",
        "候选4；K=5",
        "HA10006、HA10036、HA10057、HA10156",
        "0.4853 / 0.4509 / 0.4277 / 0.4124",
        "均为“符合你的偏好”，反馈项为0",
    ])
    fill_row(comparison_table, 2, [
        "反馈后",
        "同用户、同请求、同候选",
        "候选4；K=5；点赞HA10006",
        "顺序不变；首件标注“你点赞过”",
        "0.8487 / 0.8056 / 0.7803 / 0.7574",
        "反馈项升至1.0，其他相似商品同步上升",
    ])

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：完成“多模态输入",
        [
            "端到端测试分两层。桥接层用无人物隐私的家电示意图调用/vision，验证健康检查、"
            "JSON字段、真实类目和内存图片处理；项目一层使用Flask test client模拟登录用户上传同一图片，"
            "验证视觉分析、建议查询和原有SmartSearch排序。浏览器端再逐项检查文件上传、摄像头抓拍、"
            "可编辑文本框、确认跳转、503提示与390像素窄屏布局。",
            "跨服务trace_id已经实现。Flask在API入口生成或接收X-Trace-ID，调用桥接时继续传递，"
            "桥接响应头和JSON返回同一trace；推荐请求、意图、来源、证据和错误响应均保留该字段。"
            "阶段1与阶段6结构化验收报告记录了传递结果。",
        ],
        body_template,
    )
    e2e_table = find_table(doc, "用例")
    fill_row(e2e_table, 1, [
        "E01", "文字", "文字条件→SmartSearch；图片→/api/vision/analyze→SmartSearch",
        "正常流程", "通过；200元以内视觉建议返回4条",
        "outputs/stage22/stage22_vision_recommendation.json",
    ])
    fill_row(e2e_table, 2, [
        "E02", "语音", "WAV→SenseVoiceSmall→查询文本→现有搜索",
        "正常/首次模型加载", "通过；桥接ASR自检7.1秒，识别文本非空",
        "multimodal_ai_project1/outputs/e2e_test_results.csv",
    ])
    fill_row(e2e_table, 3, [
        "E03", "图片/摄像头", "浏览器缩放→JPEG→视觉分析→用户确认",
        "上传、摄像头、窄屏", "通过；图片与摄像头均返回结构化线索；移动端无横向溢出",
        "outputs/stage22/stage22_vision_result.png；outputs/stage22/stage22_vision_mobile.png",
    ])
    fill_row(e2e_table, 4, [
        "E04", "任一", "桥接不可用、无图片、非法格式、超大图片",
        "服务不可用/边界", "通过；分别返回503/400/400/413，文字与语音页面保持可用",
        "tests/test_vision_api.py；浏览器降级验收",
    ])

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：综合分析推荐合理性",
        [
            "视觉链路在Apple M2、16GB统一内存和CPU推理下，桥接自检单次耗时5.29秒；"
            "项目一完整请求耗时11.54秒，返回“家用电器、家居装饰、智能设备”三个合法类目、"
            "三个可见物体和类目建议词。加入“200元以内”后过滤出4件家电商品，均返回价格、"
            "热度和“符合你的偏好”理由。该结果说明视觉线索可以进入真实推荐链路，"
            "但模型的类目排序仍可能把背景装饰识别为家居装饰，因此页面保留人工确认。",
            "第一个失败案例是视觉查询最初同时加入三个类目和多个颜色，导致“白色”等高频关键词"
            "把结果拉向家居装饰。检查后发现SmartSearch的关键词采用任一命中，颜色不是硬性AND条件。"
            "修复方式是默认只使用首个主类目和最多三个物体名，颜色仅在缺少物体时使用；"
            "复测后查询收敛为“200元以内 家用电器 冰箱 洗衣机 扫地机器人”，4条结果全部属于家用电器。",
            "第二个失败案例是网页multipart上传首次返回400。原因不是图片格式，而是Flask旧辅助函数"
            "先读取了包含boundary的完整表单正文，没有优先读取request.files。改为优先取上传文件后，"
            "同一张家电图成功返回200。另一次页面初始化异常定位到结果区不存在时仍写device-info，"
            "已增加空节点保护并通过浏览器复测。",
            "当前限制是视觉模型只做单图、单次理解，不分析连续视频，也不读取商品标签文字；"
            "CPU下单次仍需数秒到十余秒；边缘-tts在2026-10-21复测时遇到上游HTTP 403，"
            "说明播报依赖外部服务，提交演示前必须重新验证。上述限制不影响文字、语音识别与推荐主链。"
        ],
        body_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明代码目录",
        [
            "项目一与项目二分别使用hygc_qzj（Python 3.11）和multimodal_ai（Python 3.10）环境。"
            "启动前确认Ollama已拉取qwen2.5:3b与qwen2.5vl:3b。先从项目二目录运行"
            "“python bridge_server.py”，再从项目根目录运行“python web/app.py”，"
            "浏览器访问http://127.0.0.1:5000。访问/health可查看multimodal.vision状态；",
            "复现视觉回归时，在项目一环境运行scripts/18_check_vision_recommend.py；"
            "桥接自检运行multimodal_ai_project1/scripts/10_test_bridge.py。"
            "若桥接未启动，视觉面板显示503说明；若视觉模型缺失，根据提示执行ollama pull qwen2.5vl:3b。"
            "报告和配置不记录实际密钥，公开交付前应只通过环境变量提供密钥。"
        ],
        body_template,
    )
    delivery_table = find_table(doc, "交付物")
    delivery_rows = [
        ("可复现代码包与README", "是", "recommender_project；multimodal_ai_project1/README.md",
         "推荐、桥接、视觉、网页与测试脚本"),
        ("各阶段输出、日志与必要截图", "是", "outputs/；multimodal_ai_project1/outputs/",
         "stage01~stage22产物、测试JSON与截图"),
        ("架构图与接口说明", "是", "outputs/stage22/stage22_project3_architecture.png；第5章",
         "/vision、/api/vision/analyze及降级规则"),
        ("部署/运行说明与风险登记", "是", "第6章；项目README",
         "双进程启动、模型依赖、隐私与外部服务风险"),
        ("项目一模块报告", "是", "《行业工程实践》...报告第3章；项目一演示稿",
         "推荐链路、指标、失败案例与接口"),
        ("项目二模块报告", "是", "本报告第4章；multimodal_ai_project1/README.md",
         "多模态采集、LLM/RAG、VL/TTS与测试"),
        ("课程综合实训报告及演示材料", "是", "本报告；项目一演示稿与既有演讲材料",
         "三项目整合、视觉选品、证据索引"),
    ]
    for row_index, values in enumerate(delivery_rows, start=1):
        fill_row(delivery_table, row_index, list(values))

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：结合课程大纲",
        [
            "系统处理的是教学演示与合成交易数据，仍按真实数据风险设计控制。视觉图片只在内存和"
            "本地HTTP请求中使用，桥接端重新编码去除EXIF，不保存原图；提示词禁止识别人物身份、"
            "年龄、性别和健康信息。账号密码使用PBKDF2加盐哈希，会话密钥与模型密钥不写入报告，"
            "反馈数据库只保留用户名、商品ID和动作。",
            "推荐公平方面，模型会偏好流行商品，并可能把图片背景误判为商品类目，因此页面展示"
            "识别依据和过滤步骤，允许用户修改且保留人工反馈。外部数据集、模型和代码在参考资料中"
            "注明来源；合成数据明确标注为生成数据。CPU推理带来能耗与响应时间限制，当前用手动单图"
            "识别和超时降级控制资源占用。"
        ],
        body_template,
    )
    risk_table = find_table(doc, "风险/影响维度")
    risk_rows = [
        ("隐私与数据安全", "图片可能包含人脸、证件或宿舍环境",
         "隐私泄露、误识别人物", "本地处理、8MB上限、重编码去EXIF、不落盘、禁止人物识别",
         "接口代码与临时文件检查"),
        ("密钥与服务安全", "配置文件或日志可能带密钥", "凭据泄露",
         "报告不引用实际密钥；优先环境变量；日志不打印图片和base64", "代码检查与.gitignore"),
        ("知识产权/引用规范", "外部数据、模型和开源库来源不清", "署名缺失或误用",
         "在参考文献列出来源；合成数据单独标注", "第9章与数据说明"),
        ("算法偏差与推荐公平", "热门商品占优；视觉背景影响类目", "长尾商品曝光不足、误推荐",
         "保留用户反馈、展示解析条件、允许确认修改、记录限制", "无结果/失败案例与反馈测试"),
        ("法律/文化/社会影响", "未经同意采集他人画面或语音", "侵犯隐私或造成误解",
         "仅使用本人同意场景；公开演示避开敏感信息；不推断身份", "页面隐私提示与人工验收"),
        ("环境与可持续发展", "本地视觉模型CPU推理耗时和内存较高", "功耗增加、设备门槛",
         "手动触发、单图、缩放、超时和503降级", "耗时与资源记录"),
    ]
    for row_index, values in enumerate(risk_rows, start=1):
        fill_row(risk_table, row_index, list(values))

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：说明项目中有证据支撑",
        [
            "可验证的改进包括：跨conda环境的本地HTTP桥接；视觉模型只从真实42个类目中选择"
            "category_hints；用户确认后才搜索；单图内存重编码且不落盘；桥接失败时返回503并保留"
            "文字与语音主链；统一的原图、视觉建议、模型、耗时和Top-K证据文件。",
            "局限包括视觉模型无法读取商品标签或连续视频，CPU推理不适合实时逐帧推荐，"
            "类目提示仍可能受背景干扰，反馈没有回流训练精排模型，TTS依赖外部网络。"
            "后续可加入OCR、连续帧稳定性和本地离线TTS，并把反馈数据用于精排训练；"
            "在引入这些能力前，报告不把计划功能写成已完成。"
        ],
        body_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（本部分每名学生必须独立填写",
        [
            "本人负责推荐系统、项目二核心模块和项目三整合的主要实现：完成数据清洗、"
            "ItemCF/Item2Vec、召回融合、LR/LightGBM精排、Flask接口、反馈闭环和推荐首页；"
            "完成多模态助手的语音、LLM/RAG、视觉、TTS与界面集成；在项目三新增视觉桥接、"
            "图片上传、查询合成、网页确认和测试证据。"
        ],
        body_template,
    )
    personal_table = find_table(doc, "序号")
    personal_rows = [
        ("1", "推荐系统全流程", "清洗、召回、排序、服务与反馈", "src/、scripts/、web/", "100%"),
        ("2", "多模态助手核心能力", "ASR、VLM、LLM/RAG、TTS与界面", "multimodal_ai_project1/", "100%"),
        ("3", "项目三视觉接入", "桥接/vision、Flask代理、查询合成与网页", "bridge_server.py、vision.js", "100%"),
        ("4", "测试和文档", "阶段日志、端到端测试、报告与演示材料", "outputs/、README、报告", "100%"),
    ]
    for row_index, values in enumerate(personal_rows, start=1):
        fill_row(personal_table, row_index, list(values))

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：选择本人最能",
        [
            "核心实现一是推荐服务。SmartSearch把规则/大模型解析出的条件转换成DataFrame过滤，"
            "再将用户向量相似度、对数热度与反馈相似度加权排序；输入接口包含q、用户和force_llm，"
            "输出包含constraints、Top-K与每条理由。该结构使视觉、语音和文字入口可以复用同一排序核心。",
            "核心实现二是项目二桥接。bridge_server把ASR、手势、视觉、LLM与TTS放在独立进程，"
            "用单把推理锁串行化模型调用。视觉入口接收base64图片和合法类目，先在内存中缩放重编码，"
            "再调用Qwen2.5-VL并解析JSON。输出经过字段收敛后回传，模型或服务失败不会让进程崩溃。",
            "核心实现三是图片到推荐的安全衔接。visual_intent模块把模型结果限制到真实类目，"
            "默认只用主类目和物体名合成查询；预算由用户在可编辑搜索框单独输入，"
            "视觉结果不携带虚假价格。前端vision.js负责拍照、选图、浏览器缩放、取消、预览和确认，"
            "图片只在本地服务请求中使用。",
        ],
        body_template,
    )

    # Replace the text placeholder table under 7.2 with the mobile verification image.
    caption_7 = find_para(doc, "图7-1")
    before_7 = caption_7._p.getprevious()
    if before_7 is not None and before_7.tag == qn("w:tbl"):
        before_7.getparent().remove(before_7)
    replace_paragraph(caption_7, "图7-1  视觉选品模块的个人实现与移动端验收")
    insert_picture_before(
        doc, caption_7, VISION_MOBILE, 2.35,
        None,
        body_template, caption_template,
    )

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：列出本人负责部分",
        [
            "交付目录现为Git工作区，本次验收不虚构提交哈希。工作证据以01至06可重跑脚本及其"
            "JSON、JSONL、CSV与Markdown产物为准；项目二性能来自performance_metrics.csv与"
            "e2e_test_results.csv；真实视觉证据来自stage22，阶段6用确定性桥接替身覆盖故障降级。"
        ],
        body_template,
    )
    evidence_table = find_table(doc, "证据类型")
    evidence_rows = [
        ("Git工作区", "main / 2026-10-24", "当前工作树", "版本追溯", "模型与敏感文件已移出Git跟踪"),
        ("六阶段验收", "2026-10-24", "scripts/01–06", "契约/会话/编排/解释/反馈/联调", "结构化产物逐阶段落盘"),
        ("自动化测试", "2026-10-24", "tests/；multimodal_ai_project1/tests/", "协议/会话/证据/编排/桥接", "根项目与桥接项目均通过"),
        ("提交包", "2026-10-24", "dist/project3_submission.zip", "最终交付", "显式清单排除模型与敏感数据"),
    ]
    for row_index, values in enumerate(evidence_rows, start=1):
        fill_row(evidence_table, row_index, list(values))

    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：至少选取",
        [
            "问题一：首次视觉查询把多个类目与白色、蓝色等高频颜色一起写入查询，推荐结果被"
            "家居装饰占据。排查发现query_parser的关键词是任一命中，而颜色在大量商品名中重复。"
            "解决为默认只使用首个主类目和最多三个物体名，颜色仅在无物体时参与，并在界面展示全部线索。"
            "用家电示意图复测，过滤后4件商品全部属于家用电器。",
            "问题二：网页multipart上传有效图片却返回400。逐步检查发现Flask辅助函数先读取原始表单正文，"
            "得到的是boundary和字段头，并非JPEG文件。改为优先读取request.files中的文件内容后，"
            "同一请求返回200，视觉结果和推荐跳转均通过浏览器复测。"
        ],
        body_template,
    )
    issue_table = find_table(doc, "问题")
    fill_row(issue_table, 1, [
        "视觉查询被高频颜色稀释",
        "200元以内家电图返回家居装饰",
        "关键词采用任一命中；颜色在商品名中高频出现",
        "默认只合入主类目和物体；颜色改作展示证据",
        "复测4件结果均为家用电器",
    ])
    fill_row(issue_table, 2, [
        "multipart图片被误判格式",
        "浏览器上传有效JPEG返回400",
        "_binary_body先读完整form data而非文件",
        "优先读取request.files，再回退raw body",
        "接口200，网页预览与推荐成功",
    ])

    summary_texts = [
        "这次实训让我第一次把推荐系统从一份交易表做到可以被文字、语音和图片调用的页面。"
        "项目一里最有价值的认识不是某个算法名称，而是每个指标都必须有清楚的比较口径。"
        "早期负样本从全库按热度采样，训练AUC只有0.3658，模型把热门程度学成了负向信号；"
        "改成与推理一致的召回候选池后，训练和评测才重新可比。召回上界0.1385和精排"
        "Precision@10 0.0715的差距也让我明白，调排序前必须先判断真实商品有没有进入候选池。",
        "项目二让我接触到语音、图像、手势和大模型在同一条界面链路中的协作。VAD断句、"
        "16k单声道格式、线程间信号、模型懒加载和退出回收这些看似琐碎的约束，决定了现场演示是否稳定。"
        "视觉模型在CPU上单次需要二十秒左右，知识库回答却可以在两秒内完成，我因此学会按延迟给能力定位，"
        "而不是把“多模态”当成一个没有边界的标签。",
        "项目三的难点是把两个已经各自可运行的项目连起来，同时不破坏原来的接口。直接import会碰到"
        "同名config包和相互冲突的依赖，所以最终采用本地HTTP桥接。桥接服务只增加能力端点，"
        "不改变推荐排序；视觉结果先变成可核验的场景、物体和合法类目，再由用户确认。"
        "这种设计牺牲了一次自动点击，却换来了可解释、可纠错和更容易测试的交互。",
        "调试视觉查询时，我第一次直观看到“模型说对了部分内容”也可能带来坏结果。模型正确识别了冰箱、"
        "洗衣机和浅绿色背景，但把家居装饰也列进类目，并把白色写进查询；而现有搜索对关键词采用任一匹配，"
        "结果很快偏到装饰品。最后的修正并没有换更大的模型，而是重新划分职责：视觉负责看图，规则负责"
        "合法类目和查询收敛，排序继续使用原有SmartSearch，用户保留最终确认权。",
        "工程规范方面，固定随机种子、按时间切分、统一输出命名、健康检查、分级错误码和日志证据"
        "让问题可以从终端追到具体文件。隐私方面，我在视觉入口增加原图内存处理、EXIF重编码和人物"
        "身份限制；报告里不记录密钥，也不把含人脸的抓拍当作成果截图。测试中TTS遇到外部HTTP 403时，"
        "我没有把失败写成通过，而是记录为待复测风险，这比展示一条漂亮但不可信的结果更重要。",
        "接下来我会继续补OCR和连续帧稳定性，让视觉能读取真实商品标签；把反馈回流到精排训练；"
        "继续增加跨服务分段延迟监控，并准备本地离线TTS，降低外部服务对演示的影响。"
        "这次实训最大的收获是把“能演示”推进到“能解释、能复现、能降级”，也让我更愿意在写代码之外，"
        "认真记录每个设计选择和失败原因。",
    ]
    replace_prompt_with_paragraphs(
        doc,
        "（填写提示：结合4周、96学时",
        summary_texts,
        body_template,
    )

    reference_prompt = find_para(doc, "（填写提示：列出本报告")
    reference_prompt._p.getparent().remove(reference_prompt._p)
    reference_anchor = find_para(doc, "[1] 作者.")
    for prefix in ("[2] 项目/模型官方文档", "[3] 数据集/开源代码来源"):
        paragraph = find_para(doc, prefix)
        paragraph._p.getparent().remove(paragraph._p)
    references = [
        "[1] 范忆梅, 林雪铭. 2627A《行业工程实践》2025级智能科学与技术（专升本）实验指导书[Z]. 2026.",
        "[2] UCI Machine Learning Repository. Online Retail[DB/OL]. https://archive.ics.uci.edu/dataset/352/online+retail, 2026-10-21.",
        "[3] Qwen Team. Qwen2.5-VL[EB/OL]. https://github.com/QwenLM/Qwen2.5-VL, 2026-10-21.",
        "[4] Ollama. Ollama API and model library[EB/OL]. https://github.com/ollama/ollama, 2026-10-21.",
        "[5] FunAudioLLM. SenseVoice[EB/OL]. https://github.com/FunAudioLLM/SenseVoice, 2026-10-21.",
        "[6] Google. MediaPipe Hand Landmarker[EB/OL]. https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker, 2026-10-21.",
        "[7] LightGBM Developers. LightGBM documentation[EB/OL]. https://lightgbm.readthedocs.io/, 2026-10-21.",
        "[8] Pallets Projects. Flask documentation[EB/OL]. https://flask.palletsprojects.com/, 2026-10-21.",
    ]
    for ref in references:
        reference_anchor._p.addprevious(copy_paragraph(body_template, ref))
    reference_anchor._p.getparent().remove(reference_anchor._p)

    replace_prompt_with_paragraphs(
        doc,
        "（可附接口 JSON",
        [
            "视觉分析请求使用multipart/form-data，字段image为JPEG或PNG文件，query为用户已有条件。"
            "下列示例省略登录Cookie，仅展示业务字段。",
        ],
        body_template,
    )
    appendix_anchor = find_para(
        doc, "附录B、测试日志与结构化输出索引", skip_toc=True)
    code_template = body_template
    code_lines = [
        "POST /api/vision/analyze",
        "Content-Type: multipart/form-data",
        "image=<家电示意图.jpg>",
        "query=200元以内",
        "",
        "200 OK",
        '{"vision":{"scene":"室内有冰箱和洗衣机",',
        '  "objects":["冰箱","洗衣机"],"colors":["白色","蓝色"],',
        '  "category_hints":["家用电器"],"search_query":"家用电器"},',
        '  "suggested_query":"200元以内 家用电器 冰箱 洗衣机",',
        '  "model":"qwen2.5vl:3b","elapsed_ms":11538}',
        "",
        "字段说明：category_hints只允许来自商品课程类目；suggested_query可编辑；",
        "图片只在本机服务间传输，不写入outputs或数据库。",
    ]
    insert_code_before(appendix_anchor, code_lines, code_template)

    replace_prompt_with_paragraphs(
        doc,
        "（列出 outputs",
        [
            "表A-1按正文位置列出可供复查的结构化文件、字段位置和测试备注。"
            "路径均相对于项目根目录；日志中的数值如与后续重跑不一致，以固定种子重跑后的新日志为准。",
        ],
        body_template,
    )

    trace_table = find_table(doc, "正文位置/指标")
    trace_rows = [
        ("5.1~5.8 视觉闭环", "outputs/stage22/stage22_vision_recommendation.json",
         "image、vision、suggested_query、recommendation", "2026-10-21实测"),
        ("3.5 推荐指标", "outputs/stage06/stage06_comparison.csv；outputs/stage05/stage05_*",
         "Precision/Recall/AUC", "SEED=42"),
        ("3.6 首页自检", "outputs/stage14/stage14_home_check.txt", "模块层与路由层PASS项", "11+9项"),
        ("4.7 多模态性能", "multimodal_ai_project1/outputs/performance_metrics.csv；e2e_test_results.csv",
         "延迟、内存、用例状态", "含2项人工长时用例"),
    ]
    for row_index, values in enumerate(trace_rows, start=1):
        fill_row(trace_table, row_index, list(values))

    # Ensure no template prompts or placeholders survive.
    leftovers = [
        p.text for p in doc.paragraphs
        if "（填写提示" in p.text or "在此插入" in p.text
        or "（列出 outputs" in p.text or "（可附接口" in p.text
        or p.text.strip().startswith("[1] 作者.")
    ]
    if leftovers:
        raise RuntimeError("Template placeholders remain: " + " | ".join(leftovers[:5]))

    # Keep cover metadata on page one and captions with their following tables/figures.
    for label in (
        "学年学期：", "课程名称：", "课程代码：", "项目名称：", "专    业：",
        "班    级：", "组    别：", "姓    名：", "学    号：", "指导教师：",
        "总    分：",
    ):
        paragraph = find_para(doc, label)
        paragraph.paragraph_format.line_spacing = 1.30
    for paragraph in doc.paragraphs:
        if paragraph.style.name == "图表题注":
            paragraph.paragraph_format.keep_with_next = True

    width_map = {
        "需求编号": [0.65, 0.80, 2.45, 1.75],
        "项目模块": [1.15, 2.55, 1.95],
        "成员": [0.70, 0.90, 2.25, 1.35, 0.55],
        "配置项": [1.35, 4.25],
        "软件/组件": [1.15, 1.85, 2.60],
        "项目": [1.15, 4.45],
        "推荐策略": [0.85, 2.05, 0.75, 2.00],
        "测试场景": [0.90, 1.35, 1.90, 1.45],
        "用例编号": [0.55, 0.85, 1.20, 1.10, 1.20, 0.75],
        "指标": [0.95, 1.45, 1.35, 1.85],
        "接口/模块": [1.05, 1.25, 1.35, 0.90, 1.15],
        "对照项": [0.65, 1.25, 0.85, 1.15, 0.80, 1.00],
        "用例": [0.55, 0.72, 1.25, 1.00, 0.93, 1.25],
        "交付物": [1.20, 0.70, 2.00, 1.70],
        "风险/影响维度": [1.10, 1.25, 1.10, 1.40, 0.85],
        "序号": [0.45, 1.10, 1.70, 1.70, 0.65],
        "证据类型": [0.85, 0.85, 1.55, 0.90, 1.45],
        "问题": [1.15, 1.05, 1.15, 1.25, 1.05],
        "正文位置/指标": [1.25, 1.80, 1.50, 1.00],
    }
    for header, widths in width_map.items():
        try:
            set_table_widths(find_table(doc, header), widths)
        except ValueError:
            # Template variants may omit optional tables; required tables were filled above.
            continue
    for table in doc.tables:
        keep_table_rows_together(table)

    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(("[1]", "[2]", "[3]", "[4]",
                                              "[5]", "[6]", "[7]", "[8]")):
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT
            paragraph.paragraph_format.left_indent = Inches(0.35)
            paragraph.paragraph_format.first_line_indent = Inches(-0.35)
            paragraph.paragraph_format.line_spacing = 1.15
            for run in paragraph.runs:
                run.font.name = "Times New Roman"
                run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), "宋体")
                run.font.size = Pt(10.5)

    update_update_fields(doc)
    doc.core_properties.title = "《行业工程实践》专升本学生实训报告"
    doc.core_properties.subject = "多模态个性化推荐助手综合实训报告"
    doc.core_properties.comments = "项目一、项目二与项目三已完成，依据实际代码与测试输出整理。"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
