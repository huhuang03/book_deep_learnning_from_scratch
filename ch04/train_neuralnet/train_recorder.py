import json
import os
import sqlite3
from typing import Optional
from typing import TypedDict

import numpy as np


class TrainRecord(TypedDict):
    index: int
    w1: np.ndarray
    b1: np.ndarray
    w2: np.ndarray
    b2: np.ndarray
    loss: float
    accuracy: float

def create_table_if_need():
    c, db = get_cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS train_records (
        id INTEGER PRIMARY KEY,
        _index INTEGER UNIQUE,
        w1 TEXT NOT NULL,
        b1 TEXT NOT NULL,
        w2 TEXT NOT NULL,
        b2 TEXT NOT NULL,
        lose REAL NOT NULL,
        accuracy REAL NOT NULL
    )
    ''')
    db.commit()
    c.close()
    db.close()

def insert(item: TrainRecord):
    c, db = get_cursor()
    v = lambda src: json.dumps(src.tolist())
    c.execute('''
    INSERT OR IGNORE INTO train_records (_index, w1, b1, w2, b2, lose, accuracy)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (item['index'],
          v(item['w1']),
          v(item['b1']),
          v(item['w2']),
          v(item['b2']),
          item['loss'],
          item['accuracy']))
    db.commit()
    c.close()
    db.close()

def find_by_index(index: int) -> Optional[TrainRecord]:
    c, db = get_cursor()
    c.execute('''
    SELECT * from train_records
    WHERE _index = ?
    ''', (index,))
    row = c.fetchone()
    v = lambda src: np.array(json.loads(src))
    if row:
        (index, w1_bytes, b1_bytes, w2_bytes, b2_bytes, loss, accuracy) = row[1:]
        return {
            'w1': v(w1_bytes),
            'b1': v(b1_bytes),
            'w2': v(w2_bytes),
            'b2': v(b2_bytes),
            'index': index,
            'loss': loss,
            'accuracy': accuracy
        }
    db.commit()
    c.close()
    db.close()

def update_accuracy(index: int, accuracy: float):
    c, db = get_cursor()

    c.execute('''
    UPDATE train_records
    SET accuracy = ?
    WHERE _index = ?
    ''', (accuracy, index))
    db.commit()
    c.close()
    db.close()


def get_cursor():
    cur_dir = os.path.dirname(__file__)
    sqlite3_db_name = 'train_record.db'
    sqlite3_db_path = os.path.join(cur_dir, sqlite3_db_name)
    db = sqlite3.connect(sqlite3_db_path)
    c = db.cursor()
    return c, db


create_table_if_need()