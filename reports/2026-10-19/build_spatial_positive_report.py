"""Build a self-contained report for one frozen spatial-positive experiment."""
import argparse
import json
import hashlib
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

parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path);args=parser.parse_args()
ROOT=args.root
R=json.loads((ROOT/'run.json').read_text());S=json.loads((ROOT/'results_summary.json').read_text())
V=json.loads((ROOT/'completion_verification.json').read_text());P=json.loads((ROOT/'protocol.json').read_text())
assert V['all_checks_passed'] and R['evaluation_status']=='COMPLETE'
official_path=ROOT/'official_results.json'
assert hashlib.sha256(official_path.read_bytes()).hexdigest()==P['official_results']['sha256']
OFFICIAL=json.loads(official_path.read_text())
pdfmetrics.registerFont(TTFont('SpatialCN','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('SpatialCN',normal='SpatialCN',bold='SpatialCN',italic='SpatialCN',boldItalic='SpatialCN')
NAVY=colors.HexColor('#183A55');W=A4[0]-36*mm
ST={n:ParagraphStyle(n,fontName='SpatialCN',fontSize=z,leading=l,wordWrap='CJK',spaceAfter=8,
    textColor=NAVY if n in ('title','heading') else colors.black)
    for n,z,l in [('title',20,28),('heading',12,20),('body',10,16),('note',8.5,13),('cell',9,13)]}
ST['head']=ParagraphStyle('head',fontName='SpatialCN',fontSize=9,leading=13,textColor=colors.white)
story=[]
def add(t,s='body'):story.append(Paragraph(t,ST[s]))
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(v)),ST['head']) for v in headers]]
    data += [[Paragraph(escape(str(v)),ST['cell']) for v in row] for row in rows]
    t=Table(data,colWidths=[W*w for w in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F7')]),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,8)])
def fmt(split,arm,metric='F1'):
    v=S['summary'][split][arm][metric];digits=4 if metric=='F1' else 2
    mean=f"{v['mean']:.{digits}f}"
    return mean if v['sd'] is None else mean+f" ± {v['sd']:.{digits}f}"
labels={'single':'原单正样本InfoNCE','multi':'空间多正样本InfoNCE'}
seeds=S['evaluated_seeds']
add('RLPR：空间多正样本对照','title')
add('以验证集F1选优 · 网络/温度/训练预算固定 · 后台自动执行','note')
add('本轮选定：'+labels[S['selected_family']]+'。对比的跨模态种子：'+', '.join(map(str,seeds))+'。每个新增模型训练1轮，固定原LiDAR/Radar第5轮权重。')
add('一、验证集筛选与选优','heading')
gate=json.loads((ROOT/'screening.json').read_text())
add('seed0筛选'+('通过，随后完成seed1/2复验。' if S['promoted'] else '未通过，按预定规则停止后续种子训练并保留原方法。'))
table(['方法','Sejong F1','Sejong R@1（%）'],[[labels[a],fmt('sejong',a),fmt('sejong',a,'Recall@1')] for a in labels],[.44,.3,.26])
add('筛选先比较seed0验证F1，再比较R@1；只有严格改善才追加另外两个种子。最终按相同种子集合的平均验证F1、平均R@1、完全平分时原方法优先选模。所有新增测试均在锁定选择之后进行，不选择测试最高的单个种子。','note')
add('二、测试结果与论文、官方模型比较','heading')
rows=[]
for split,name in [('dcc','DCC'),('riverside','Riverside')]:
    old=OFFICIAL['splits'][split]
    rows += [[name,'论文RLPR',f"{old['paper_metrics']['F1']:.4f}",f"{old['paper_metrics']['Recall@1']:.2f}"],
             [name,'官方权重本机测试',f"{old['metrics']['F1']:.4f}",f"{old['metrics']['Recall@1']:.2f}"]]
    rows += [[name,labels[a],fmt(split,a),fmt(split,a,'Recall@1')] for a in labels]
table(['数据集','方法','F1','R@1（%）'],rows,[.16,.37,.26,.21])
add(('均值±SD为三个跨模态种子的均值及样本标准差。' if len(seeds)==3 else '本轮仅完成seed0筛选；表中实验值为单次结果，不代表多种子均值或稳定提升。')+'固定单模态权重，不能解释为完整训练流程的多种子实验。','note')
add('三、逐种子F1','heading')
table(['seed','DCC 原方法','DCC 多正样本','Riv. 原方法','Riv. 多正样本'],
    [[s]+[f"{(R['baseline'] if a=='single' else R['trials'])[str(s)]['benchmarks'][split]['F1']:.4f}" for split in ('dcc','riverside') for a in labels] for s in seeds],[.1,.225,.225,.225,.225])

story.append(PageBreak())
add('实验定义与核验','title')
add('四、本轮唯一实验变量','heading')
add('原InfoNCE在每组12帧中进行同步实例的对角匹配。新方法使用相同12帧的训练坐标：距离≤9米的配对为正样本，距离>12米为负样本，9至12米区域忽略。每行正样本概率相加，使用其负对数作为损失；同步雷达教师到LiDAR学生的方向保持一致。')
add('L = mean[logsumexp(valid logits) - logsumexp(positive logits)]','note')
add('只使用训练帧的坐标；坐标按原采样函数实际读取帧的顺序记录，不额外抽样。没有引入跨组负样本。原网络、评估器、数据划分及历史源文件均保持不变。')
table(['固定条件','设置'],[
    ['初始化与轮数','固定原幅度版L5/R5；每个新模型从相同单模态权重开始跨模态第1轮，不从旧跨模态模型续训'],
    ['训练参数','温度0.07，batch8，Adam 5e-5，5困难+5随机负样本，严格确定性执行'],
    ['辅助项','相位辅助0.1；融合监督0，融合前向和BN路径仍保留；局部/全局描述子都使用相同多正样本规则'],
    ['数据与指标','MulRan训练：Sejong训练区域+KAIST；验证：Sejong独立区域；测试：DCC/Riverside；最大F1沿用原测试脚本的1000个阈值扫描'],
],[.25,.75])
add('五、核验结果','heading')
add('新坐标传递流程的单正样本控制组已完整训练，模型张量与历史seed0 InfoNCE模型逐字节一致，Sejong验证指标也一致。新增多正样本模型均检查冻结雷达教师、有限权重/优化器、2071次更新和完整训练设置。')
add(f"已核验配对采样随机流、实际困难负样本、源码/协议/模型/评估记录哈希及选优锁。完整回归检查{V['regression_tests']}项通过。报告由已完成的JSON结果自动生成，不改变任何分数。",'note')
add('六、解释范围','heading')
add('这是在已经查看过DCC/Riverside结果之后进行的探索性对照。验证筛选通过不保证测试提升；筛选失败也不会按测试成绩重新启用候选。若只运行seed0，须继续把结果标为单次筛选。尚未验证其他数据集与4D雷达泛化。')
add('七、结果文件','heading')
add('目录：'+escape(str(ROOT))+'<br/>protocol.json：冻结条件；screening.json：筛选；selection.json：最终选择<br/>results_summary.json、comparison.csv：完整结果；completion_verification.json：核验<br/>各模型：control/weights、multi/seed*/weights；逐查询诊断：各模型目录CSV/NPZ','note')
STAMP=datetime.now(timezone(timedelta(hours=8)))
def footer(c,doc):
    c._doc.info._dateFormatter=lambda *a:STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('SpatialCN',8);c.drawString(18*mm,12*mm,'RLPR | 空间多正样本 | '+STAMP.strftime('%Y-%m-%d'))
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(ROOT/'report.pdf'),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
    title='RLPR空间多正样本对照',author='RLPR实验记录').build(story,onFirstPage=footer,onLaterPages=footer)
print(ROOT/'report.pdf')
