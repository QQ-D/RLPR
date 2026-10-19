"""Export verified final amplitude-ablation results without altering training sources."""
import json
from pathlib import Path
from datetime import datetime, timezone, timedelta
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'runs/mulran_word_amplitude_official_2026-10-18_193110'
R = json.loads((ROOT / 'run.json').read_text())
S = json.loads((ROOT / 'results_summary.json').read_text())
V = json.loads((ROOT / 'completion_verification.json').read_text())
assert R['status'] == 'COMPLETE' and S['evidence_verified'] and V['all_checkpoints_verified'] == 22
OUT = HERE / 'RLPR修改版最终实验结果汇报_2026-10-19.pdf'
STAMP = datetime.now(timezone(timedelta(hours=8)))
pdfmetrics.registerFont(TTFont('CN', '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', subfontIndex=0))
pdfmetrics.registerFontFamily('CN', normal='CN', bold='CN', italic='CN', boldItalic='CN')
NAVY = colors.HexColor('#183A55')
W = A4[0] - 36 * mm
styles = {
    'title': ParagraphStyle('title', fontName='CN', fontSize=21, leading=29, textColor=NAVY, spaceAfter=12),
    'heading': ParagraphStyle('heading', fontName='CN', fontSize=13, leading=21, textColor=NAVY, spaceBefore=12, spaceAfter=7),
    'body': ParagraphStyle('body', fontName='CN', fontSize=10, leading=16, wordWrap='CJK', spaceAfter=8),
    'note': ParagraphStyle('note', fontName='CN', fontSize=8.3, leading=13, textColor=colors.HexColor('#536474'), wordWrap='CJK', spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='CN', fontSize=9, leading=13, wordWrap='CJK'),
    'head': ParagraphStyle('head', fontName='CN', fontSize=9, leading=13, textColor=colors.white),
}
story = []
def add(t, style='body'):
    story.append(Paragraph(t, styles[style]))
def table(headers, rows, widths):
    data = [[Paragraph(escape(str(x)), styles['head']) for x in headers]]
    data += [[Paragraph(escape(str(x)), styles['cell']) for x in row] for row in rows]
    t = Table(data, colWidths=[W*x for x in widths], repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, colors.HexColor('#F0F5F8')]),
        ('TOPPADDING',(0,0),(-1,-1),7), ('BOTTOMPADDING',(0,0),(-1,-1),7),
        ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#D8E2E9')),
    ]))
    story.extend([t,Spacer(1,8)])
labels = {'paper':'论文 RLPR', 'released_official_model':'官方权重本机测试',
          'pure_phase':'修改版：纯相位', 'amplitude':'修改版：保留幅度'}
def rows(split):
    return [[labels[x['method']]] + [f"{x[k]:.4f}" if k=='F1' else f"{x[k]:.2f}" for k in ('Recall@1','Recall@5','Recall@10','F1')]
            for x in S['comparison'] if x['dataset']==split and x['method'] in labels]

add('RLPR 修改版最终实验结果汇报', 'title')
add('MulRan · 纯相位与保留实测频谱幅度的受控对照', 'note')
add('训练与测试完成：2026年10月19日 03:49（北京时间）。两组均完成 LiDAR 5轮、Radar 5轮、跨模态1轮；本报告中的成绩均为雷达查询到 LiDAR 数据库的检索结果。')
add('一、最终测试成绩', 'heading')
add('DCC（数据库204帧，查询188帧）', 'heading')
table(['模型','R@1 (%)','R@5 (%)','R@10 (%)','F1'], rows('dcc'), [.36,.16,.16,.16,.16])
add('Riverside（数据库1558帧，查询1724帧）', 'heading')
table(['模型','R@1 (%)','R@5 (%)','R@10 (%)','F1'], rows('riverside'), [.36,.16,.16,.16,.16])
add('二、可以得出的结论', 'heading')
add('1. 在本轮严格配对实验中，保留幅度相对纯相位的 R@1：DCC 提高4.26个百分点，Riverside 提高8.93个百分点；F1分别提高0.0146、0.0710。')
add('2. 保留幅度版 DCC R@1 为55.32%，达到论文及官方权重本机测试水平；Riverside 为62.82%，低于论文2.03个百分点，低于官方本机测试2.44个百分点。尚未超过官方模型。')
add('3. 此前按验证集选定的困难负样本版 Riverside R@1为56.09%；本次提高至62.82%。该历史比较同时改变了初始化和训练设置，不能把全部提升都归因于幅度。')
add('本轮只有一个随机种子，属于探索性结果；需要多种子复验才能判断提升的稳定性。DCC与Riverside此前已被查看，不属于全新盲测。', 'note')

story.append(PageBreak())
add('实验设置、核验与后续建议', 'title')
add('三、两组共同的训练设置', 'heading')
table(['项目','设置'], [
    ['训练数据','Sejong01/02训练区域 + KAIST01/02；数据库16602帧，查询16568帧'],
    ['验证数据','Sejong独立地理区域；数据库1505帧，查询1380帧；训练与验证索引分离，约50米地理缓冲'],
    ['初始化与轮数','同seed0与ResNet18 ImageNet骨干初始化；各自从头训练总计5/5/1轮；固定使用单模态第5轮与跨模态第1轮'],
    ['优化与采样','batch 8；Adam，学习率5e-5，每轮乘0.8；5个困难负样本+5个随机负样本；负样本距离大于12米'],
    ['修改模块','频域相位Transformer、动态频率掩码、局部/全局描述子、训练时相位交叉注意力、signed_coherency_v2对齐'],
    ['唯一实验变量','纯相位，或在相位自注意力及训练时交叉注意力后恢复实测FFT幅度；评估时两路独立提取描述子'],
], [.24,.76])
add('轮数来自作者代码默认值。论文本身没有明确写出MulRan训练轮数，因此不将5/5/1表述为论文明确给定值。', 'note')
add('四、完成后的证据核验', 'heading')
add('已逐一核验22个检查点：文件哈希与记录一致，参数及优化器状态均为有限值，单模态每轮2071个优化步骤。两组数据采样随机流一致；两组跨模态阶段的雷达教师参数均保持冻结。源代码、协议、初始化权重和选模记录哈希检查通过；DCC/Riverside测试发生在两组选模锁定后。')
add('五、后续建议', 'heading')
add('优先把保留幅度版作为下一轮候选，冻结本次代码与权重。先做多种子复验，再在同一设置下单独消融交叉融合、频率掩码或对齐损失；每次只改变一个因素。若继续调参，依据验证集选择，避免按DCC/Riverside结果挑选轮数。')
add('当前结果支持“保留幅度值得继续研究”，尚不足以证明提出的方法优于RLPR。暂不追加训练，也不把增加轮数直接等同于性能提升。', 'note')
add('六、结果与权重位置', 'heading')
add('实验目录：RLPR/runs/mulran_word_amplitude_official_2026-10-18_193110<br/>保留幅度权重：amplitude/cross/weights/1.pth.tar<br/>纯相位权重：pure_phase/cross/weights/1.pth.tar<br/>对比表：comparison.csv；曲线：curves.csv<br/>结果记录：run.json、results_summary.json<br/>核验记录：completion_verification.json', 'note')
add('本报告范围为MulRan；未宣称完成全部论文数据集或4D雷达泛化验证。', 'note')
def decorate(c, doc):
    c._doc.info._dateFormatter = lambda *args: STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('CN',8);c.setFillColor(colors.HexColor('#536474'))
    c.drawString(18*mm,12*mm,'RLPR | 最终受控对照 | 2026-10-19')
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
                  title='RLPR修改版最终实验结果汇报',author='RLPR实验记录').build(story,onFirstPage=decorate,onLaterPages=decorate)
(HERE/'report_data.json').write_text(json.dumps({'summary':S,'verification':V,'finished_at':R['finished_at']},ensure_ascii=False,indent=2)+'\n')
print(OUT)
