#Read cvs file to seprate PD and HC image datas

import os
import sys
import csv
import shutil

### Read labels from csv files
def split_label(csv_file):
    lable_file = open(csv_file,'r')
    reader = csv.reader(lable_file)
    header = next(reader)

    pic_Index = header.index('Subject')
    group_Index = header.index('Group')

    for row in reader:
        if row[group_Index] == 'PD':
            PD.append( row[pic_Index])
        elif row[group_Index] == 'Control':
            HC.append( row[pic_Index])
        else:
            print("Unknown group")
    
    lable_file.close()

### Seg specific slice for each subject from the original dataset
def labeled_img (labeled_list, ori_dir, new_dir):
    for num in labeled_list:
        for item in os.listdir(ori_dir):
            if num in item:
                
                input_f = os.path.join(ori_dir, item)
                ex_command = "med2image -i %s -d ./%s -o mwp1%s.jpg -s m" %(input_f,new_dir,num)
                os.system(ex_command)
                
                #shutil.copy(os.path.join(ori_dir, item),new_dir)
                # use this line if you don't want to keep the original data in the original GM/WM folder
                # shutil.copy(os.path.join(ori_dir, item),new_dir) 


# Change path !!!
file_Path = "/Users/DXX/Desktop/UACLASS/MM811/project/Datas"
GM_path = "/Users/DXX/Desktop/UACLASS/MM811/project/Datas/GM"
WM_PATH = "/Users/DXX/Desktop/UACLASS/MM811/project/Datas/WM"

# Change target directory
gm_PD_folder = 'gmPD_slice' 
gm_HC_folder = 'gmHC_slice' 
wm_PD_folder = 'wmPD_slice' 
wm_HC_folder = 'wmHC_slice'

csv_file = 'PPMI.csv'
PD=[]
HC=[]
split_label(csv_file)

labeled_img(HC,WM_PATH,wm_HC_folder)
labeled_img(PD,WM_PATH,wm_PD_folder)
labeled_img(HC,GM_PATH,gm_HC_folder)
labeled_img(PD,GM_PATH,gm_PD_folder)

