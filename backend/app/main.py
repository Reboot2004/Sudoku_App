from __future__ import annotations
import json, sqlite3
from datetime import date
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/'data'; DB=DATA/'sudoku.db'; DATA.mkdir(exist_ok=True)
app=FastAPI(title='DC Sudoku API',version='0.4.0')
app.add_middleware(CORSMiddleware,allow_origins=['*'],allow_methods=['*'],allow_headers=['*'])
class PersonalPuzzle(BaseModel):
    title:str='My Sudoku'
    grid:list[list[int]]
def validate(grid):
    if len(grid)!=9 or any(len(r)!=9 for r in grid): raise HTTPException(422,'Grid must be exactly 9x9.')
    if any(v<0 or v>9 for r in grid for v in r): raise HTTPException(422,'Values must be 0..9.')
def conflict(grid):
    for r in range(9):
        v=[x for x in grid[r] if x]
        if len(v)!=len(set(v)): return True
    for c in range(9):
        v=[grid[r][c] for r in range(9) if grid[r][c]]
        if len(v)!=len(set(v)): return True
    for br in range(0,9,3):
        for bc in range(0,9,3):
            v=[grid[r][c] for r in range(br,br+3) for c in range(bc,bc+3) if grid[r][c]]
            if len(v)!=len(set(v)): return True
    return False
def init_db():
    with sqlite3.connect(DB) as con:
        con.execute('CREATE TABLE IF NOT EXISTS personal_puzzles (id INTEGER PRIMARY KEY AUTOINCREMENT,title TEXT NOT NULL,grid_json TEXT NOT NULL,created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)')
init_db()
@app.get('/health')
def health(): return {'status':'ok'}
@app.get('/puzzles/dc/today')
def dc_today():
    p=DATA/'today.json'
    return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {'date':str(date.today()),'edition':'Hyderabad','source':'Deccan Chronicle','puzzles':[]}
@app.post('/puzzles/personal',status_code=201)
def create_personal(payload:PersonalPuzzle):
    validate(payload.grid)
    if conflict(payload.grid): raise HTTPException(422,'Puzzle contains row, column, or box conflicts.')
    with sqlite3.connect(DB) as con:
        cur=con.execute('INSERT INTO personal_puzzles(title,grid_json) VALUES (?,?)',(payload.title,json.dumps(payload.grid)))
    return {'id':cur.lastrowid,'title':payload.title,'grid':payload.grid}
@app.get('/puzzles/personal')
def list_personal():
    with sqlite3.connect(DB) as con: rows=con.execute('SELECT id,title,grid_json,created_at FROM personal_puzzles ORDER BY id DESC').fetchall()
    return [{'id':r[0],'title':r[1],'grid':json.loads(r[2]),'created_at':r[3]} for r in rows]
