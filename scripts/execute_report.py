"""在仓库目录内执行 Notebook，每个单元格完成后保存进度。"""
from pathlib import Path
import time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
path=ROOT/'Store_Sales.ipynb'
n=nbformat.read(path,as_version=4)
start=time.time()
def done(cell,cell_index,**kwargs):
    nbformat.write(n,path)
    if cell.cell_type=='code':
        print('完成单元格',cell_index,'累计秒数',round(time.time()-start,1),flush=True)
        for o in cell.outputs:
            if o.output_type=='stream':print(o.text[-1200:],flush=True)
client=NotebookClient(n,timeout=1800,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}},on_cell_executed=done)
try:client.execute()
finally:nbformat.write(n,path)
print('全部执行成功',flush=True)
