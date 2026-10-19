"""Render the repository's research plan as a native-text Chinese PDF."""
import argparse
from datetime import datetime,timezone,timedelta
from html import escape
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--markdown',type=Path,default=HERE.parents[1]/'github_export/docs/NEXT_STEPS.md')
parser.add_argument('--output',type=Path,default=HERE/'RLPR后续研究方案_2026-10-19.pdf')
args=parser.parse_args()
pdfmetrics.registerFont(TTFont('PlanCN','/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',subfontIndex=0))
pdfmetrics.registerFontFamily('PlanCN',normal='PlanCN',bold='PlanCN',italic='PlanCN',boldItalic='PlanCN')
NAVY=colors.HexColor('#183A55');W=A4[0]-36*mm
ST={name:ParagraphStyle(name,fontName='PlanCN',fontSize=size,leading=leading,wordWrap='CJK',spaceAfter=5,
    keepWithNext=name in ('title','heading'),textColor=NAVY if name in ('title','heading') else colors.black)
    for name,size,leading in [('title',20,28),('heading',12,20),('body',9.5,14),('cell',8.5,12)]}
ST['head']=ParagraphStyle('head',fontName='PlanCN',fontSize=8.5,leading=12,textColor=colors.white)
story=[]
def clean(text):
    text=re.sub(r'\[([^\]]+)\]\([^)]+\)',r'\1',text)
    return escape(text.replace('**','').replace('`',''))
def paragraph(text,style='body'):story.append(Paragraph(clean(text),ST[style]))
lines=args.markdown.read_text().splitlines();i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            parts=[p.strip() for p in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?',p) for p in parts):rows.append(parts)
            i+=1
        n=len(rows[0]);widths=[.52,.16,.16,.16] if n==4 else [.12,.44,.44] if n==3 else [1/n]*n
        data=[[Paragraph(clean(v),ST['head'] if ri==0 else ST['cell']) for v in row] for ri,row in enumerate(rows)]
        table=Table(data,colWidths=[W*w for w in widths],repeatRows=1)
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#EFF4F7')]),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([table,Spacer(1,9)]);continue
    if line.startswith('# '):paragraph(line[2:],'title')
    elif line.startswith('## '):paragraph(line[3:],'heading')
    else:
        text=line
        while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].startswith(('#','|','- ')) and not re.match(r'\d+\. ',lines[i+1]):
            i+=1;text+=' '+lines[i].strip()
        paragraph(text)
    i+=1
STAMP=datetime.now(timezone(timedelta(hours=8)))
def footer(c,doc):
    c._doc.info._dateFormatter=lambda *a:STAMP.strftime("D:%Y%m%d%H%M%S+08'00'")
    c.setFont('PlanCN',8);c.drawString(18*mm,12*mm,'RLPR | 后续研究与实验方案 | 2026-10-19')
    c.drawRightString(A4[0]-18*mm,12*mm,str(doc.page))
SimpleDocTemplate(str(args.output),pagesize=A4,leftMargin=18*mm,rightMargin=18*mm,topMargin=16*mm,bottomMargin=20*mm,
    title='RLPR后续研究与实验方案',author='RLPR实验记录').build(story,onFirstPage=footer,onLaterPages=footer)
print(args.output)
