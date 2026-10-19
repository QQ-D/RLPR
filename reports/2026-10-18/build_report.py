"""Build a Chinese PDF from completed RLPR evidence; training is read-only."""
import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
STAMP = datetime.now(timezone(timedelta(hours=8)))
PDF = HERE / 'RLPR复现与改进阶段结果汇报_2026-10-18.pdf'
CSV = PROJECT / 'runs/mulran_word_hardneg_2026-10-16_003626/all_results_comparison.csv'
with CSV.open(encoding='utf-8-sig') as handle:
    ROWS = list(csv.DictReader(handle))
OFFICIAL = json.loads((PROJECT / 'runs/mulran_official_eval_2026-10-14/results.json').read_text())
DIAG = json.loads((PROJECT / 'runs/mulran_teacher_diagnostic_2026-10-16_013559/results.json').read_text())
CURRENT = PROJECT / 'runs/mulran_word_amplitude_official_2026-10-18_193110'
state_bytes = (CURRENT / 'run.json').read_bytes()
STATE = json.loads(state_bytes)
PROTOCOL = json.loads((CURRENT / 'protocol.json').read_text())
assert PROTOCOL['schedule'] == 'official_code'
assert PROTOCOL['unimodal_epochs'] == 5 and PROTOCOL['cross_epochs'] == 1

pdfmetrics.registerFont(TTFont('CN', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN', italic='CN', boldItalic='CN')
NAVY = colors.HexColor('#183A55')
TEAL = colors.HexColor('#087D86')
GRAY = colors.HexColor('#536474')
LIGHT = colors.HexColor('#EDF3F7')
W = A4[0] - 34 * mm
styles = {
    'title': ParagraphStyle('title', fontName='CN', fontSize=24, leading=33, textColor=NAVY, spaceAfter=8),
    'sub': ParagraphStyle('sub', fontName='CN', fontSize=9.2, leading=14, textColor=GRAY, spaceAfter=9),
    'h1': ParagraphStyle('h1', fontName='CN', fontSize=16.5, leading=24, textColor=NAVY, spaceAfter=10),
    'h2': ParagraphStyle('h2', fontName='CN', fontSize=11.4, leading=18, textColor=TEAL, spaceBefore=9, spaceAfter=5),
    'body': ParagraphStyle('body', fontName='CN', fontSize=9.7, leading=15.5, spaceAfter=6, wordWrap='CJK'),
    'note': ParagraphStyle('note', fontName='CN', fontSize=8.4, leading=12.5, textColor=GRAY, spaceAfter=6, wordWrap='CJK'),
    'cell': ParagraphStyle('cell', fontName='CN', fontSize=8.5, leading=12.2, wordWrap='CJK'),
    'num': ParagraphStyle('num', fontName='CN', fontSize=8.7, leading=12, alignment=TA_CENTER),
    'head': ParagraphStyle('head', fontName='CN', fontSize=8.5, leading=12.2, textColor=colors.white, wordWrap='CJK'),
}
story = []


def para(text, style='body'):
    return Paragraph(text, styles[style])


def add(text, style='body'):
    story.append(para(text, style))


def title(number, text):
    add(f'{number:02d}  {text}', 'h1')


def table(headers, rows, widths, *, highlights=()):
    data = [[para(escape(str(v)), 'head') for v in headers]]
    for row in rows:
        data.append([para(escape(str(v)), 'cell' if i == 0 else 'num') for i, v in enumerate(row)])
    t = Table(data, colWidths=[W * v for v in widths], hAlign='LEFT', repeatRows=1)
    commands = [('BACKGROUND', (0, 0), (-1, 0), NAVY),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('LEFTPADDING', (0, 0), (-1, -1), 6), ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('LINEBELOW', (0, 0), (-1, 0), .6, NAVY),
                ('LINEBELOW', (0, 1), (-1, -1), .35, colors.HexColor('#D8E2E9'))]
    for i in range(1, len(data)):
        if i % 2 == 0:
            commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor('#F6F8FA')))
    for i in highlights:
        commands.append(('BACKGROUND', (0, i + 1), (-1, i + 1), colors.HexColor('#E5F2EF')))
    t.setStyle(TableStyle(commands))
    story.extend([t, Spacer(1, 6)])


def lookup(split, model, epoch=None):
    candidates = [r for r in ROWS if r['split'] == split and r['model'] == model
                  and (epoch is None or r['epoch'] == str(epoch))]
    assert len(candidates) == 1, (split, model, epoch)
    return candidates[0]


def metric(row, name):
    return f"{float(row[name]):.4f}" if name == 'F1' else f"{float(row[name]):.2f}"


def benchmark_row(label, split, model, epoch=None):
    r = lookup(split, model, epoch)
    return [label] + [metric(r, k) for k in ('Recall@1', 'Recall@5', 'Recall@10', 'F1')]


def new_page():
    story.append(PageBreak())


add('RLPR 复现与改进<br/>阶段结果汇报', 'title')
add('MulRan 完整划分 · 已完成实验结果与当前训练计划', 'sub')
add(f'汇报快照：{STAMP:%Y年%m月%d日 %H:%M}（北京时间）<br/>范围：官方模型复现、Word 方案改造、受控对照与频谱幅度实验。', 'sub')
summary = Table([[para('阶段结论：官方预训练模型已基本复现论文水平；修改模型在 DCC 的部分指标接近官方，但 Riverside 仍有明显差距。当前新启动的“纯相位 / 保留幅度”对照尚无最终跨模态测试成绩。')]], colWidths=[W])
summary.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, -1), LIGHT),
                             ('BOX', (0, 0), (-1, -1), .6, colors.HexColor('#CCDDE8')),
                             ('LEFTPADDING', (0, 0), (-1, -1), 11), ('RIGHTPADDING', (0, 0), (-1, -1), 11),
                             ('TOPPADDING', (0, 0), (-1, -1), 10), ('BOTTOMPADDING', (0, 0), (-1, -1), 9)]))
story.extend([summary, Spacer(1, 12)])
title(1, '目前最值得汇报的结果')
table(['结果来源', 'DCC R@1', 'Riverside R@1', '用途'], [
    ['论文 RLPR', '55.32%', '64.85%', '目标参考'],
    ['官方权重本机测试', '55.32%', '65.26%', '已完成的复现基准'],
    ['修改模型：困难负样本，第2轮', '55.32%', '56.09%', '最新按验证集规则选定版本'],
    ['历史 InfoNCE 描述子损失对照', '54.79%', '57.48%', 'Riverside 历史较高结果，未被验证集选中'],
], [.35, .15, .21, .29], highlights=(2,))
add('上述 Recall@1 均为雷达查询到 LiDAR 数据库的检索成功率。最新选定版本在 Riverside 比官方本机基准低9.17个百分点；历史 InfoNCE 对照仍低7.78个百分点。', 'note')
add('已经完成', 'h2')
add('已完成 MulRan 所需8条序列的工程处理与主测试划分评估；独立单模态预训练、非对称跨模态训练和测试均可运行；已保存完整权重、日志、协议与对照结果。频域改造、训练时交叉融合和检查点配置校验已接入。')
add('本报告的边界', 'h2')
add('历史较高数值不等于一个统一的最优模型，也不替代验证集选模。当前结果来自单种子探索性实验；本报告聚焦 MulRan，不能据此宣称已完成所有数据集或4D雷达泛化验证。')
add('论文：RLPR: Radar-to-LiDAR Place Recognition via Two-Stage Asymmetric Cross-Modal Alignment for Autonomous Driving，IEEE Robotics and Automation Letters，2026。', 'note')

new_page()
title(2, '工程、数据划分与方法改动')
add('数据与训练/测试协议', 'h2')
table(['划分', '数据与区域', '数据库 / 查询'], [
    ['训练', 'Sejong01/02训练区域 + KAIST01/02', '16,602 / 16,568'],
    ['验证', 'Sejong独立地理区域', '1,505 / 1,380'],
    ['DCC测试', 'DCC01数据库 / DCC02查询', '204 / 188'],
    ['Riverside测试', 'Riverside01数据库 / Riverside02查询', '1,558 / 1,724'],
], [.18, .56, .26])
add('Sejong训练和验证使用地理划分，保留约50米缓冲；已核查索引无同帧混入。元信息文件含测试帧或脚本预生成测试BEV，不等于这些帧参加梯度训练。扫描感知范围为80米，因此不把“索引分离”表述为感知视野完全无重叠。', 'note')
table(['项目', '设置'], [
    ['输入与检索', 'BEV高50×宽225，范围80米；描述子512维；真值阈值9米'],
    ['训练样本', '正样本距离≤9米；负样本排除12米内候选；batch size 8'],
    ['优化', 'Adam，初始学习率5×10⁻⁵，每轮乘0.8'],
    ['运行环境', 'Conda rlpr；PyTorch 2.1 / CUDA 12.1；RTX 4060 Laptop 8GB'],
], [.25, .75])
add('已实现的修改模块', 'h2')
table(['模块', '当前实现与作用'], [
    ['频域表示', 'FFT相位表示、可学习动态频率掩码；频域相位Transformer后再iFFT。'],
    ['描述子与对齐', '局部/全局描述子各256维；雷达分支冻结；已修正有符号相关度对齐损失，并保留InfoNCE对照。'],
    ['交叉融合', '训练时 LiDAR 查询雷达的相位交叉注意力；评估时两路独立提取描述子。'],
    ['采样与辅助项', '频域辅助损失和融合损失可控；训练集内支持5个困难+5个随机负样本。'],
    ['新增幅度对照', '保留测得FFT幅度，在自注意力和交叉融合后恢复幅度；相位Transformer输入保持一致。'],
], [.24, .76])
add('Word文字要求丢弃幅度，而示例代码保留“幅度×掩码”。目前把两种解释做成独立开关验证；尚未证明幅度保留一定有效。代码完整回归测试79项通过，含频谱边界、冻结雷达、配置匹配与恢复检查。', 'note')

new_page()
title(3, '论文、官方复现与代表性修改结果')
add('指标：R@K 为 Recall@K（%）；F1 使用当前统一的原代码PR评估口径。绿色行为最新按验证集选定的修改版本。历史实验的随机流与阶段设置存在差异，不能把跨行差值全部归因于单一模块。', 'note')
methods = [('论文 RLPR', '论文RLPR', None), ('官方权重本机测试', '官方权重本机测试', None),
           ('历史 Word 初始完整模型', '历史Word旧MSC', None),
           ('历史 Word 相关度修正', '历史Word修正相关度', None),
           ('历史 InfoNCE 对照（未选定）', '仅替换描述子损失为InfoNCE', None),
           ('困难负样本，第2轮（选定）', '5困难+5随机采样', 2)]
for split, heading in [('dcc', 'DCC：跨模态检索'), ('riverside', 'Riverside：跨模态检索')]:
    add(heading, 'h2')
    table(['模型 / 设置', 'R@1', 'R@5', 'R@10', 'F1'],
          [benchmark_row(label, split, model, epoch) for label, model, epoch in methods],
          [.40, .15, .15, .15, .15], highlights=(5,))
add('如何理解“最接近”', 'h2')
add('官方权重本机结果与论文最接近：DCC的R@1相同，Riverside高0.41个百分点。修改模型的历史DCC最高R@1为55.85%（相关度修正版本）；Riverside历史最高为57.48%（InfoNCE或关闭频域辅助损失的对照）。这些最高值来自不同实验，不能拼成同一个模型成绩。')
add('本次阶段汇报以“困难负样本第2轮”作为最新规则选定版本，同时完整列出历史较高对照。其选择依据是Sejong验证R@1，其次F1、再取较早轮次；没有用DCC/Riverside来重选该轮次。', 'note')

new_page()
title(4, '轮数与困难负样本：已完成的受控证据')
add('验证集曲线：跨模态阶段从1轮训练到5轮', 'h2')
curve_rows = []
for e in range(1, 6):
    a = lookup('sejong_validation', '严格随机采样', e)
    b = lookup('sejong_validation', '5困难+5随机采样', e)
    curve_rows.append([e, metric(a, 'Recall@1'), metric(a, 'F1'), metric(b, 'Recall@1'), metric(b, 'F1')])
table(['跨模态轮次', '随机 R@1', '随机 F1', '困难采样 R@1', '困难采样 F1'],
      curve_rows, [.18, .19, .19, .22, .22], highlights=(1,))
add('随机采样组按验证规则选第4轮；困难采样组选第2轮（与第5轮R@1相同，但F1更高）。训练loss持续下降，没有带来测试性能持续提高。此前随机组Riverside R@1从第1轮55.86%降至第5轮53.07%；困难组从56.55%降至53.77%。', 'note')
add('相同跨模态轮次的比较：均为第2轮', 'h2')
table(['测试集 / 采样', 'R@1', 'R@5', 'R@10', 'F1'], [
    benchmark_row('DCC / 随机', 'dcc', '严格随机采样', 2),
    benchmark_row('DCC / 5困难+5随机', 'dcc', '5困难+5随机采样', 2),
    benchmark_row('Riverside / 随机', 'riverside', '严格随机采样', 2),
    benchmark_row('Riverside / 5困难+5随机', 'riverside', '5困难+5随机采样', 2),
], [.40, .15, .15, .15, .15], highlights=(1, 3))
add('结论与适用范围', 'h2')
add('同轮次下，困难采样使DCC R@1提高1.06个百分点、Riverside提高0.64个百分点；部分R@5/R@10/F1未同步提高。因此可认为有小幅收益，不能表述为所有指标全面提升。')
add('两组使用相同单模态起始权重、种子、批次预算和优化设置；困难样本只从训练集已有的12米外负候选中挖掘，每轮刷新。匹配轮次的比较在各自选择规则锁定后执行，未据测试成绩重选模型。')
add('轮数口径要区分', 'h2')
add('本页1—5轮指“跨模态对齐阶段”的轮数；当前新任务则是单模态各5轮、跨模态1轮。它们不是同一种训练预算，也不把旧权重上的追加微调重新记作从头训练。', 'note')

new_page()
title(5, '瓶颈判断、正在运行的实验与证据索引')
add('单模态诊断：Riverside的表示能力也有差距', 'h2')
diagnostic_rows = []
for item in DIAG['results']:
    if item['split'] == 'riverside':
        label = '官方模型' if item['name'] == 'official' else '修改模型：困难采样第2轮'
        diagnostic_rows.append([label] + [f"{item['metrics'][k]['Recall@1']:.2f}" for k in ('R2L', 'R2R', 'L2L')])
table(['模型', '雷达到LiDAR', '雷达到雷达', 'LiDAR到LiDAR'], diagnostic_rows, [.40, .20, .20, .20])
add('R2R反映冻结雷达分支，L2L使用跨模态训练后的LiDAR分支；单模态数值不是跨模态性能的严格上限。该结果提示应同时检查表示能力与对齐机制，尚不能断言“幅度丢失”就是唯一原因。', 'note')
add('当前实验：保持修改代码，按官方代码的轮数比较两种表示', 'h2')
table(['项目', '当前安排'], [
    ['两组方法', '纯相位 / 保留幅度；其余模型、数据与训练设置一致'],
    ['初始化与轮数', '相同seed0与ResNet18初始化；LiDAR5轮、Radar5轮、跨模态1轮'],
    ['跨模态输入权重', '固定使用各组单模态第5轮；验证指标仅作诊断'],
    ['最终评估', '两组完成后测试DCC、Riverside，比较官方基准及组间差值'],
], [.25, .75])
phase = escape(str(STATE.get('phase', '准备中')))
add(f'快照状态：{escape(STATE["status"])}；阶段 {phase}。新对照的最终跨模态成绩尚未产生。论文未明确给出MulRan轮数，上述5＋5＋1来自官方代码默认值。', 'note')
add('数据来源与复核入口（相对 RLPR 工程根目录）', 'h2')
evidence = [
    ('[1] 论文与官方权重基准', 'runs/mulran_official_eval_2026-10-14/results.json'),
    ('[2] 全部历史对照数值', 'runs/mulran_word_hardneg_2026-10-16_003626/all_results_comparison.csv'),
    ('[3] 同轮次困难采样比较', 'runs/mulran_word_hardneg_2026-10-16_003626/matched_epoch_results.json'),
    ('[4] 单模态诊断', 'runs/mulran_teacher_diagnostic_2026-10-16_013559/results.json'),
    ('[5] 当前运行协议与日志', 'runs/' + CURRENT.name + '/protocol.json；run.json'),
]
for label, path in evidence:
    add(escape(label) + '：' + escape(path), 'note')
add('本报告附report_data.json保存数值与训练状态快照；后续正式报告将用两组训练完成后的新成绩更新本阶段结论。', 'note')


def decorate(canvas, doc):
    canvas._doc.info._dateFormatter = lambda *args: STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor('#CBD7DF'))
    canvas.line(17 * mm, 18 * mm, A4[0] - 17 * mm, 18 * mm)
    canvas.setFont('CN', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(17 * mm, 13 * mm, 'RLPR 复现与改进｜阶段性结果，当前幅度对照仍在训练')
    canvas.drawRightString(A4[0] - 17 * mm, 13 * mm, f'{doc.page} / 5')
    canvas.restoreState()


document = SimpleDocTemplate(str(PDF), pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
                             topMargin=17 * mm, bottomMargin=23 * mm,
                             title='RLPR复现与改进阶段结果汇报（2026-10-18）', author='RLPR项目实验记录',
                             subject='MulRan官方复现、修改模型对照与当前幅度实验')
document.build(story, onFirstPage=decorate, onLaterPages=decorate)
data = dict(generated_at=STAMP.isoformat(), scope='MulRan completed evidence; new official-budget amplitude comparison pending',
            completed_metrics=ROWS, official=OFFICIAL, sensor_diagnostic=DIAG,
            current_run_snapshot=STATE, current_protocol=PROTOCOL,
            current_snapshot_sha256=hashlib.sha256(state_bytes).hexdigest(),
            sources={str(CSV.relative_to(PROJECT)): hashlib.sha256(CSV.read_bytes()).hexdigest()})
(HERE / 'report_data.json').write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
print(PDF)
