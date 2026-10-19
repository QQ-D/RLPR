"""Render verified phase-auxiliary weight comparisons after completion."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from datetime import datetime, timezone, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, PageBreak

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / 'runs/mulran_amplitude_phase_2026-10-19_130536'
R = json.loads((ROOT/'run.json').read_text())
S = json.loads((ROOT/'results_summary.json').read_text())
V = json.loads((ROOT/'completion_verification.json').read_text())
assert R['status'] == 'COMPLETE' and V['all_checks_passed']
OUT = HERE / 'RLPR相位辅助损失消融结果_2026-10-19.pdf'
STAMP = datetime.now(timezone(timedelta(hours=8)))
selected = S['selected_configuration']
weight = R['configurations'][selected]['weight']
pdfmetrics.registerFont(TTFont('PhaseCN','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('PhaseCN',normal='PhaseCN',bold='PhaseCN',italic='PhaseCN',boldItalic='PhaseCN')
NAVY = colors.HexColor('#183A55')
W = A4[0] - 36*mm
styles = {name:ParagraphStyle(name,fontName='PhaseCN',fontSize=size,leading=leading,
          spaceAfter=8,textColor=NAVY if name in ('title','heading') else colors.black,wordWrap='CJK')
          for name,size,leading in [('title',21,29),('heading',13,21),('body',10,16),('note',8.5,13),('cell',9,13)]}
styles['head'] = ParagraphStyle('head',fontName='PhaseCN',fontSize=9,leading=13,textColor=colors.white)
story=[]
def add(text,style='body'):
    story.append(Paragraph(text,styles[style]))
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(x)),styles['head']) for x in headers]]
    data += [[Paragraph(escape(str(x)),styles['cell']) for x in row] for row in rows]
    t=Table(data,colWidths=[W*x for x in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F7')]),
       ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),
       ('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#D8E2E9'))]))
    story.extend([t,Spacer(1,8)])
labels={'paper':'论文 RLPR','released_official':'官方权重本机测试',
        'phase_01':'幅度版：相位辅助权重0.1','phase_003':'幅度版：相位辅助权重0.03','phase_00':'幅度版：相位辅助权重0'}
def metric_rows(split):
    return [[labels[x['configuration']] + ('（选定）' if x['selected'] else '')] +
            [f"{x[k]:.4f}" if k=='F1' else f"{x[k]:.2f}" for k in ('Recall@1','Recall@5','Recall@10','F1')]
            for x in S['comparisons'] if x['dataset']==split]

add('RLPR 相位辅助损失消融结果', 'title')
add('固定融合监督0 · 固定单模态权重 · 只改变相位辅助损失权重', 'note')
add(f"完成时间：{R['finished_at'][:19].replace('T',' ')}（北京时间）。验证集选定：相位辅助权重 {weight:g}。本轮比较的是雷达查询到LiDAR数据库的检索性能。")
add('一、Sejong验证结果与预先锁定的选择', 'heading')
table(['相位辅助权重','R@1 (%)','F1','是否选定'],
      [[f"{v['weight']:g}",f"{v['validation']['Recall@1']:.2f}",f"{v['validation']['F1']:.4f}",'是' if a==selected else '否']
       for a,v in R['configurations'].items()], [.3,.25,.25,.2])
add('全部验证完成后，按R@1、F1、配置顺序0.1/0.03/0选择并锁定，再开始DCC/Riverside测试。下表展示全部配置，不按测试成绩重新选模。', 'note')
add('二、DCC测试', 'heading')
table(['模型','R@1 (%)','R@5 (%)','R@10 (%)','F1'],metric_rows('dcc'),[.4,.15,.15,.15,.15])
add('三、Riverside测试', 'heading')
table(['模型','R@1 (%)','R@5 (%)','R@10 (%)','F1'],metric_rows('riverside'),[.4,.15,.15,.15,.15])
for split,label,paper,official in [('dcc','DCC',55.32,55.32),('riverside','Riverside',64.85,65.26)]:
    score=R['configurations'][selected]['benchmarks'][split]['Recall@1']
    base=R['configurations']['phase_01']['benchmarks'][split]['Recall@1']
    add(f"{label}：选定配置R@1为{score:.2f}%，相对0.1控制组变化{score-base:+.2f}个百分点，相对论文{score-paper:+.2f}个百分点，相对官方权重本机测试{score-official:+.2f}个百分点。", 'note')
if selected == 'phase_01':
    add('本轮结论：降低或关闭相位辅助损失没有提高验证集及两项测试的R@1，继续保留相位辅助权重0.1。其他指标有少量混合变化，不能说所有指标都下降。', 'note')

story.append(PageBreak())
add('实验设置与核验', 'title')
add('四、受控实验设置', 'heading')
table(['项目','本轮设置'],[
    ['起点','复用上轮幅度版LiDAR第5轮、Radar第5轮；三个配置分别从相同初始化开始跨模态第1轮，不累积续训'],
    ['变量','相位辅助损失权重0.1、0.03、0，同时缩放相位一致性和频谱相干性两项辅助损失；所有前向计算保持不变'],
    ['固定条件','seed0，batch8，Adam，学习率5e-5；融合监督权重0但前向计算及BN路径保留；signed_coherency_v2；5困难+5随机负样本；雷达教师冻结'],
    ['数据','训练Sejong01/02训练区域+KAIST01/02；Sejong独立地理区域作验证；DCC01/02及Riverside01/02作测试'],
    ['范围','本次只执行相位辅助损失权重消融，没有同时改变融合监督权重、对齐损失类型或训练轮数'],
],[.23,.77])
add('五、核验结果', 'heading')
add('0.1控制组完整训练后，模型张量与上一轮选定的融合监督权重0模型逐字节相同，Sejong验证结果也相同。已核验三组检查点、冻结教师、训练设置、2071次优化更新、协议与源码哈希；已比较保存的困难负样本索引和数据库/查询帧ID；运行环境与启动记录一致。')
add('共89项回归测试通过，包括实际参数解析、训练/验证目录隔离、选模规则、中断锁定恢复和检查点审计。原来的网络、训练器及已完成的实验均保持不变。', 'note')
add('六、结论的适用范围', 'heading')
add('本轮是单种子探索性消融。验证成绩的小幅差异不代表稳定提升，测试成绩也不能反过来决定采用哪个配置。即使某项指标高于论文，仍需多种子复验、同等训练预算及更多数据集来支持方法优势。DCC/Riverside已经被查看，不再属于全新盲测。')
add('本轮仅改变一个超参数，保留全部旧模型与日志；未追加其他未报告的训练。进一步实验应先固定本轮验证选出的相位辅助权重，再单独对比对齐损失，并通过多种子复验判断稳定性。')
add('七、可追溯文件', 'heading')
add('实验目录：RLPR/runs/mulran_amplitude_phase_2026-10-19_130536<br/>协议：protocol.json；运行：run.json；锁定选择：selection.json<br/>结果：results_summary.json、comparison.csv<br/>核验：completion_verification.json；测试日志：regression_tests.log<br/>各配置权重：phase_01、phase_003、phase_00目录下weights/1.pth.tar', 'note')
def decorate(c,doc):
    c._doc.info._dateFormatter=lambda *args:STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('PhaseCN',8)
    c.drawString(18*mm,12*mm,'RLPR | 相位辅助损失消融 | 2026-10-19')
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
    title='RLPR相位辅助损失消融结果',author='RLPR实验记录').build(story,onFirstPage=decorate,onLaterPages=decorate)
(HERE/'phase_report_data.json').write_text(json.dumps({'summary':S,'verification':V,'run':R},ensure_ascii=False,indent=2)+'\n')
print(OUT)
