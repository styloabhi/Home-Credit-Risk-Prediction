import asyncio

asyncio.set_event_loop_policy(
    asyncio.WindowsSelectorEventLoopPolicy()
)
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time

import subprocess

def get_notebook(default_path,title):
    notebook=Path(default_path)
    if notebook.exists():
        print(f"Using:{notebook}")
        return notebook
    root= Tk()
    root.withdraw()
    notebook=Path(askopenfilename(title=title,
                                  filetypes=[("Jupyter Notebooks","*.ipynb")]))
    return notebook

BASE_DIR=Path(__file__).parent.parent
EDA_NOTEBOOK=get_notebook(BASE_DIR/"Data Cleaning and EDA Python"/"cleaning+eda.ipynb","Select EDA Notebook")
ML_NOTEBOOK= get_notebook(BASE_DIR/"Feature Engineering and ML"/"ml+model.ipynb","Select ML Notebook")
start=time.time()
print("Running EDA Notebook...")

eda_result=subprocess.run([
    "jupyter",
    "nbconvert",
    "--execute",
    "--inplace",
    str(EDA_NOTEBOOK)
])
if eda_result.returncode!=0:
    print("EDA Notebook failed to execute")
    
print("Finished Eda")
print("Running ML Notebook...")
ml_result=subprocess.run([
    "jupyter",
    "nbconvert",
    "--execute",
    "--inplace",
    str(ML_NOTEBOOK)
])
if ml_result.returncode!=0:
    print("ML Notebook failed to execute")

print("ML Notebook Finished")
print("Pipeline Completed Successfully")
print(f"Total Time Taken: {(time.time()-start)/60:.2f} minutes")