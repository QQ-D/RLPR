"""Render the frozen three-temperature, three-cross-seed comparison."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone,timedelta
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak

parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);ROOT=parser.parse_args().root
R=json.loads((ROOT/'run.json').read_text());S=json.loads((ROOT/'results_summary.json').read_text())
P=json.loads((ROOT/'protocol.json').read_text());V=json.loads((ROOT/'completion_verification.json').read_text())
assert V['all_checks_passed'] and R['evaluation_status']=='COMPLETE'
official=ROOT/'official_results.json'
assert hashlib.sha256(official.read_bytes()).hexdigest()==P['official_results']['sha256']
O=json.loads(official.read_text());ARMS=('tau_003','tau_007','tau_010');LABELS={'tau_003':'温度0.03','tau_007':'温度0.07（原对照）','tau_010':'温度0.10'}
pdfmetrics.registerFont(TTFont('TempCN','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('TempCN',normal='TempCN',bold='TempCN',italic='TempCN',boldItalic='TempCN')
NAVY=colors.HexColor('#183A55');W=A4[0]-36*mm
ST={n:ParagraphStyle(n,fontName='TempCN',fontSize=z,leading=l,wordWrap='CJK',spaceAfter=7,textColor=NAVY if n in ('title','heading') else colors.black)
    for n,z,l in [('title',20,28),('heading',12,20),('body',10,16),('note',8.5,13),('cell',9,13)]}
ST['head']=ParagraphStyle('head',fontName='TempCN',fontSize=9,leading=13,textColor=colors.white)
story=[]
def add(text,s='body'):story.append(Paragraph(text,ST[s]))
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(v)),ST['head']) for v in headers]]
    data += [[Paragraph(escape(str(v)),ST['cell']) for v in row] for row in rows]
    t=Table(data,colWidths=[W*w for w in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F7')]),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.extend([t,Spacer(1,7)])
def fmt(split,arm,key='F1'):
    d=S['summary'][split][arm][key];n=4 if key=='F1' else 2
    return f"{d['mean']:.{n}f} ± {d['sd']:.{n}f}"
selected=S['selected_configuration']
add('RLPR：InfoNCE训练温度对照','title')
add('单正样本 · 三个温度 · 三个跨模态种子 · F1优先选优','note')
add('验证集选定：'+LABELS[selected]+'。0.07复用已完成的三个种子，0.03与0.10各新增三个种子，每个新增模型训练1轮。')
add('一、Sejong验证结果','heading')
table(['训练温度','F1（均值±SD）','R@1（%）'],[[LABELS[a]+('（选定）' if a==selected else ''),fmt('sejong',a),fmt('sejong',a,'Recall@1')] for a in ARMS],[.36,.36,.28])
add('预先固定：完成全部九项验证后，按平均F1、平均R@1、完全平分时优先原0.07选择；锁定后才开始新增模型的DCC/Riverside测试。没有按seed0筛选或挑选最高测试分数的种子。','note')
add('二、测试F1与论文、官方权重比较','heading')
rows=[]
for split,label in [('dcc','DCC'),('riverside','Riverside')]:
    old=O['splits'][split]
    rows += [[label,'论文RLPR',f"{old['paper_metrics']['F1']:.4f}",f"{old['paper_metrics']['Recall@1']:.2f}"],
             [label,'官方权重本机测试',f"{old['metrics']['F1']:.4f}",f"{old['metrics']['Recall@1']:.2f}"]]
    rows += [[label,LABELS[a],fmt(split,a),fmt(split,a,'Recall@1')] for a in ARMS]
table(['数据集','方法','F1','R@1（%）'],rows,[.16,.32,.29,.23])
add('表中实验值为三个跨模态种子的均值±样本标准差；单模态L5/R5固定来自seed0。这不能代表完整训练流程的多种子波动。最大F1沿用原评估器的1000个距离阈值扫描，未修改评估公式。','note')
add('三、配对F1变化（相对0.07）','heading')
table(['数据集','温度0.03 ΔF1','温度0.10 ΔF1'],[[label]+[f"{S['paired_F1_delta_from_007'][split][a]['mean']:+.5f}（{S['paired_F1_delta_from_007'][split][a]['positive_seed_count']}/3次提高）" for a in ('tau_003','tau_010')]
    for split,label in [('sejong','Sejong'),('dcc','DCC'),('riverside','Riverside')]],[.2,.4,.4])

story.append(PageBreak())
add('逐种子结果与实验核验','title')
add('四、逐种子测试F1','heading')
table(['seed','DCC .03','DCC .07','DCC .10','Riv. .03','Riv. .07','Riv. .10'],
    [[s]+[f"{R['configurations'][a][str(s)]['benchmarks'][split]['F1']:.4f}" for split in ('dcc','riverside') for a in ARMS] for s in (0,1,2)],[.08]+[.92/6]*6)
add('五、固定条件与实验变量','heading')
table(['项目','设置'],[
    ['唯一变量','原单正样本InfoNCE的训练温度0.03/0.07/0.10；局部/全局及融合描述子的对比损失共用该温度；保持雷达教师到LiDAR学生的方向'],
    ['训练预算','原幅度版LiDAR第5轮、Radar第5轮作为固定初始化；六个新增模型各从相同单模态权重开始跨模态第1轮，共2071次更新/模型'],
    ['其他参数','batch8，Adam 5e-5，5困难+5随机负样本，严格确定性执行；相位辅助0.1，融合监督0但融合前向及BN路径保留'],
    ['数据划分','训练Sejong训练区域+KAIST；验证Sejong独立区域；测试DCC/Riverside。原数据预处理和划分不变，不采用上一轮的多正样本掩码'],
],[.23,.77])
add('六、核验与可追溯性','heading')
add(f"已核验{V['checkpoint_count']}个跨模态检查点，其中{V['new_training_trials']}个新增训练。检查冻结雷达教师、有限模型及优化器、更新步数、实际训练温度、配对采样流与困难负样本，绑定源码、协议、模型和评估文件哈希。完整回归检查{V['regression_tests']}项通过。")
add('0.07对照复用既有模型和评估，原InfoNCE源码哈希保持一致，其默认温度为0.07。新增温度写入检查点；恢复训练要求记录温度匹配。纸面与官方对照结果在启动时复制并绑定哈希，报告只读取冻结副本。','note')
add('七、解释范围与结果文件','heading')
add('本轮是已查看过DCC/Riverside之后的探索性超参数对照。微小均值差异不足以单独证明稳定优势；应同时看各种子的配对变化和波动。验证选优不保证跨场景提升，测试结果不反向更改选择。尚未验证其他数据集或4D雷达泛化。')
add('实验目录：'+escape(str(ROOT))+'<br/>冻结设置：protocol.json；选优锁：selection.json；状态：run.json<br/>汇总：results_summary.json、comparison.csv；核验：completion_verification.json<br/>新权重：tau_003、tau_010各seed目录；报告：report.pdf、report_verification.json','note')
STAMP=datetime.now(timezone(timedelta(hours=8)))
def footer(c,doc):
    c._doc.info._dateFormatter=lambda *a:STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('TempCN',8);c.drawString(18*mm,12*mm,'RLPR | 训练温度对照 | '+STAMP.strftime('%Y-%m-%d'))
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
    title='RLPR InfoNCE训练温度对照',author='RLPR实验记录').build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'report.pdf')
