# 临时运行，将之前的训练记录存到sqlite数据库
import os.path
import sqlite3
import json

from ch04.constant import *
from ..cache import Cache, CacheItem


def _insert(cursor: sqlite3.Cursor, index, item: CacheItem):
    v = lambda src: json.dumps(src.tolist())
    cursor.execute('''
    INSERT OR IGNORE INTO train_records (_index, w1, b1, w2, b2, lose, accuracy)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (index,
          v(item.params[W1]),
          v(item.params[B1]),
          v(item.params[W2]),
          v(item.params[B2]),
          item.loss, None))

def main():
    items = Cache.load().items
    if len(items) == 0:
        print("no caches")
        return

    for item in items:
        print(item.loss)

    return

    cur_dir = os.path.dirname(__file__)
    sqlite3_db_name = 'train_record.db'
    sqlite3_db_path = os.path.join(cur_dir, sqlite3_db_name)
    db = sqlite3.connect(sqlite3_db_path)
    # create table if not exists
    c = db.cursor()
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

    for i in range(len(items)):
        _insert(c, i, items[i])

    db.commit()
    c.close()
    db.close()

if __name__ == '__main__':
    main()