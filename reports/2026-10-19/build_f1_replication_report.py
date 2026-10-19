"""Render completed F1-first replication from audited JSON artifacts."""
from pathlib import Path
import json
from xml.sax.saxutils import escape
from datetime import datetime,timezone,timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak

HERE=Path(__file__).resolve().parent
ROOT=Path(Path('/tmp/rlpr_f1_replication_run_path').read_text())
R=json.loads((ROOT/'run.json').read_text())
S=json.loads((ROOT/'results_summary.json').read_text())
V=json.loads((ROOT/'completion_verification.json').read_text())
E=json.loads((ROOT/'error_analysis.json').read_text())
assert R['status']=='COMPLETE' and V['all_checks_passed']
OFFICIAL=json.loads((ROOT.parents[1]/'runs/mulran_official_eval_2026-10-14/results.json').read_text())
OUT=HERE/'RLPR_F1优先多种子复验_2026-10-19.pdf'
STAMP=datetime.now(timezone(timedelta(hours=8)))
pdfmetrics.registerFont(TTFont('F1CN','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('F1CN',normal='F1CN',bold='F1CN',italic='F1CN',boldItalic='F1CN')
NAVY=colors.HexColor('#183A55');W=A4[0]-36*mm
styles={name:ParagraphStyle(name,fontName='F1CN',fontSize=size,leading=leading,spaceAfter=7,
    textColor=NAVY if name in ('title','heading') else colors.black,wordWrap='CJK')
    for name,size,leading in [('title',20,28),('heading',12,20),('body',10,16),('note',8.5,13),('cell',9,13)]}
styles['head']=ParagraphStyle('head',fontName='F1CN',fontSize=9,leading=13,textColor=colors.white)
story=[]
def add(t,style='body'):story.append(Paragraph(t,styles[style]))
def table(headers,rows,widths):
    data=[[Paragraph(escape(str(v)),styles['head']) for v in headers]]
    data += [[Paragraph(escape(str(v)),styles['cell']) for v in row] for row in rows]
    t=Table(data,colWidths=[W*v for v in widths],repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
       ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F7')]),
       ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    story.extend([t,Spacer(1,7)])
def ms(split,arm,metric):
    item=S['summary'][split][arm][metric]
    digits=4 if metric=='F1' else 2
    return f"{item['mean']:.{digits}f} ± {item['sd']:.{digits}f}"
labels={'signed':'Signed控制','infonce':'InfoNCE'}
add('RLPR：F1优先多种子复验','title')
add('固定单模态权重 · 跨模态阶段seed 0 / 1 / 2 · 全部结果完整报告','note')
add(f"完成时间：{R['finished_at'][:19].replace('T',' ')}（北京时间）。新增两种子×两种损失，各训练1轮；seed0复用已完成的对照。验证集选定损失：{labels[S['selected_loss']]}。")
add('一、以验证集平均F1选优','heading')
table(['损失','Sejong F1（均值±SD）','R@1（%）'],
    [[labels[a]+('（选定）' if a==S['selected_loss'] else ''),ms('sejong',a,'F1'),ms('sejong',a,'Recall@1')] for a in labels],[.3,.4,.3])
add('预先规定：先比较三个种子的Sejong平均F1，再比较平均R@1，完全相同则优先控制组。全部验证完成后锁定，随后才进行新增种子的测试；不挑选最高测试分数的单个种子。','note')
add('二、测试F1与论文、官方权重比较','heading')
rows=[]
for split,label in [('dcc','DCC'),('riverside','Riverside')]:
    p=OFFICIAL['splits'][split]
    rows.append([label,'论文RLPR',f"{p['paper_metrics']['F1']:.4f}",f"{p['paper_metrics']['Recall@1']:.2f}"])
    rows.append([label,'官方权重本机测试',f"{p['metrics']['F1']:.4f}",f"{p['metrics']['Recall@1']:.2f}"])
    rows += [[label,labels[a],ms(split,a,'F1'),ms(split,a,'Recall@1')] for a in labels]
table(['测试集','模型','F1','R@1（%）'],rows,[.17,.28,.31,.24])
add('三、逐种子测试F1','heading')
table(['种子','DCC Signed','DCC InfoNCE','Riv. Signed','Riv. InfoNCE'],
    [[s]+[f"{R['seeds'][s][a]['benchmarks'][split]['F1']:.4f}" for split in ('dcc','riverside') for a in labels] for s in ('0','1','2')],[.12,.22,.22,.22,.22])
add('均值±SD为三个跨模态种子的均值和样本标准差。固定的LiDAR/Radar第5轮权重来自seed0，因此这些数值不能代表整个训练流程的多种子波动。','note')

story.append(PageBreak())
add('F1口径、错误分析与核验','title')
add('四、F1与R@1反映不同表现','heading')
add('沿用官方脚本：先取Top-1检索结果，再按描述子距离扫描1000个阈值，报告该测试集上的最大F1。它的Recall分母是所有Top-1检索正确的查询；错误检索被拒绝记为TN。因此该F1与R@1可以呈不同趋势，不等同于所有位置查询的成功率。')
table(['seed0模型','DCC 最大F1','DCC 固定阈值F1','Riv. 最大F1','Riv. 固定阈值F1'],
    [[labels[a]]+[f"{E[split]['models'][a][key]['F1']:.4f}" for split in ('dcc','riverside') for key in ('best_threshold','validation_frozen_threshold')] for a in labels],[.22,.195,.195,.195,.195])
add('固定阈值先在Sejong验证集确定，再直接用于DCC/Riverside；这是额外的部署诊断，不能替换论文表中的测试集最大F1。','note')
add('五、seed0配对错误分析','heading')
table(['数据集','两者均正确','仅Signed正确','仅InfoNCE正确','两者均错误'],
    [[label]+[E[split]['paired_seed0'][k] for k in ('both_correct','signed_only_correct','infonce_only_correct','both_wrong')] for split,label in [('dcc','DCC'),('riverside','Riverside')]],[.2,.2,.2,.2,.2])
add('每个查询的帧ID、Top-1位置误差、描述子距离、正样本数量和首个正确结果名次，均已保存CSV；NPZ保存Top-20检索，便于继续定位误匹配。F1改善是否来自更容易区分正确/错误匹配，应结合阈值诊断判断，不能仅凭损失下降推断。','note')
add('六、训练与结果核验','heading')
add('保持幅度表示、相位辅助0.1、融合监督0（融合前向和BN路径仍保留）、batch8、Adam 5e-5、5困难+5随机负样本。每个新增模型训练2071次更新；雷达教师参数和统计量冻结。实际模型初始化种子与采样种子同时改变，配对损失使用相同随机流和困难负样本。')
add(f"已检查{V['checkpoint_count']}个跨模态检查点（其中{V['new_training_checkpoints']}个新增）、源码/协议/选优锁/初始化/评估文件哈希、教师冻结、优化步数、缓存及运行环境；{V['regression_tests_passed']}项回归测试通过。诊断F1与原评估器逐次核对一致。",'note')
add('七、结论范围与文件','heading')
add('本轮以F1为首要目标，同时保留R@1结果供判断取舍。三个种子样本较少，且DCC/Riverside在早期实验中已查看，属于探索性复验。不能仅凭单个种子的最高分宣称方法稳定超过论文；尚未验证其他数据集或4D雷达泛化。','note')
add('实验目录：'+escape(str(ROOT.relative_to(ROOT.parents[1])))+'<br/>结果：results_summary.json；选优：selection.json；核验：completion_verification.json<br/>错误诊断：error_analysis.json；逐查询CSV/NPZ位于seed0、seed1、seed2各损失目录。','note')
def decorate(c,doc):
    c._doc.info._dateFormatter=lambda *args:STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('F1CN',8);c.drawString(18*mm,12*mm,'RLPR | F1优先复验 | 2026-10-19')
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(OUT),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
    title='RLPR F1优先多种子复验',author='RLPR实验记录').build(story,onFirstPage=decorate,onLaterPages=decorate)
(HERE/'f1_replication_report_data.json').write_text(json.dumps(dict(run=R,summary=S,verification=V,error_analysis=E),ensure_ascii=False,indent=2)+'\n')
print(OUT)
