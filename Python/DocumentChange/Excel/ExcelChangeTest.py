from openpyxl import  load_workbook
import sys

wb = load_workbook('./test.xlsx')

name = sys.argv[1]
time_vec = sys.argv[2]
time_vec = time_vec.split(',')

time_v = [float(i) for i in time_vec]

sheet = wb['Blad1']


# Find row with name
for k in range(1,10):
    if sheet['A' + str(k)].value == 'Markus':
        row = str(k)


for l in range(len(time_v)):
    sheet[chr(ord('A') + l + 1) + str(row)].value = time_v[l]

wb.save('./test.xlsx')

