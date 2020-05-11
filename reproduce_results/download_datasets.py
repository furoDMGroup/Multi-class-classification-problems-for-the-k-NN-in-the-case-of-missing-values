import requests
import os.path
import setup_path as pth
from zipfile import ZipFile

datasets = [('balance-scale.data', 'https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data'),
            ('BreastTissue.xls', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00192/BreastTissue.xls'),
            ('vertebral_column_data.zip', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00212/vertebral_column_data.zip'),
            ('seeds_dataset.txt', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt'),
            ('sensor_readings_4.data', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00194/sensor_readings_4.data'),
            ('wifi_localization.txt', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00422/wifi_localization.txt'),
            ('leaf.zip', 'https://archive.ics.uci.edu/ml/machine-learning-databases/00288/leaf.zip')]

os.chdir(pth.datasets_path)
print('Started downloading data into: ' + pth.datasets_path)
for dataset in datasets:
    if os.path.isfile(dataset[0]):
        pass
    else:
        url = dataset[1]
        r = requests.get(url, allow_redirects=True)
        open(dataset[0], 'wb').write(r.content)

if not os.path.isfile('column_3C.dat'):
    with ZipFile('vertebral_column_data.zip', 'r') as zipObj:
        zipObj.extract('column_3C.dat')
if not os.path.isfile('leaf.csv'):
    with ZipFile('leaf.zip') as zipObj:
        zipObj.extract('leaf.csv')

# seeds dataset has incorrectly used separators in some lines (use of double tabs instead of one)
# we need to correct this
s = open('seeds_dataset.txt', 'r')
lines = s.readlines()
s.close()
s = open('seeds_dataset2.txt', 'w')
for l in lines:
    s.write(l.replace('\t\t', '\t'))
s.close()
os.remove('seeds_dataset.txt')
os.rename('seeds_dataset2.txt', 'seeds_dataset.txt')

print('Succesfully downloaded data and unzipped required archived data')