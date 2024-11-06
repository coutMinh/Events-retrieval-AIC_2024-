import glob
import pandas as pd
count = 0
count2 = 0
files = sorted(glob.glob('D:\database\database\*'))[1200:]

check_files = sorted(glob.glob("G:\My Drive\Colab_Notebooks\AI_challenge_2024\database\KeyFrames\*"))[1200:]
print(files)
for file in files:
    tmp = glob.glob(file + '\*.jpg')
    count += len(tmp)
for checkfile in check_files:
    tmp = glob.glob(checkfile + '\*.jpg')
    count2 += len(tmp)

print(count, count2)




check_files = [check_file.split('\\')[-1] for check_file in check_files]
files = [file.split('\\')[-1] for file in files]
df = pd.DataFrame({'file':files, 'check_file':check_files})
df.to_csv('check.csv', index=False)
for i in range(len(check_files)):
    if check_files[i] != files[i]:
        print(i)
