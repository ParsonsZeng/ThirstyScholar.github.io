#!/usr/bin/python

"""
Adapt from:
1. http://www.hoamon.info/blog/2007/02/01/pythonpdb.html

Run with command 'python3 -d pdb_practice.py' in the terimal

b 數字 - 設置中斷點
r - 繼續執行，直到當前函式返回
c - 繼續執行程式
n - 執行下一行程式
s - 進入函式
p [變數名稱] - 印出變數
l - 列出目前的程式片段
q - 離開

"""

import pdb

for i in range(5):
    j = i ** 2
    pdb.set_trace()   # enter debugger
    print('i =', i, ', j =', j)
