import csv
import glob
import os

def write_chunk_to_file(lines, file_count, output_prefix):
    output_file = f"{output_prefix}{file_count}.csv"
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(lines)

def split_csv(input_file, lines_per_file, output_prefix):
    with open(input_file, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Assuming the first line is a header
        
        lines = []
        file_count = 1
        
        for line in reader:
            lines.append(line)
            
            if len(lines) == lines_per_file:
                write_chunk_to_file(lines, file_count, output_prefix)
                lines = []
                file_count += 1
        
        if lines:
            write_chunk_to_file( lines, file_count, output_prefix)

        print(input_file, '  done!')



split_csv('/scratch2/chaoyi/DarkMachines/sm/large/single_top_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/single_top/single_top')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/single_topbar_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/single_topbar/single_topbar')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/ww_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/ww/ww')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/wtop_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/wtop/wtop')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/wtopbar_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/wtopbar/wtopbar')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/z_jets_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/z_jets/z_jets')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/ttbar_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/ttbar/ttbar')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/gam_jets_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/gam_jets/gam_jets')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/w_jets_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/w_jets/w_jets')
split_csv('/scratch2/chaoyi/DarkMachines/sm/large/njets_10fb.csv', 4700, '/scratch2/chaoyi/DarkMachines/sm/njets/njets')

path = '/scratch2/chaoyi/DarkMachines/sm/'
for i in glob.glob(path+'/*csv'):
    process = i.split('/')[-1].split('_')[0]
    os.system('mkdir '+ path +'/'+process)
    out_path = path+'/'+process+'/'+process
    print(i)
    split_csv(i, 4700, out_path)

