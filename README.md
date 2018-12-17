# MRI_Project

## Instruction

### Envirnment
    python 3.6
    Tensorflow
    numpy, matplotlib, opencv
    pandas, glob, csv, shutil, 

### Preprocessing
    1. Download all data (GM/WM folder) & PPMI.csv file
    2. change GM/WM folder path in Split_PD_HC.py 
    3. (Option) If you'd like to run the 2d CNN Model, 
                change the slice number (last number)of the [ex_command] (m stands for the middle slice)
                in Split_PD_HC.py (more detailed of using med2image can be found in https://github.com/FNNDSC/med2image )
    4. Run ## $python3 Split_PD_HC.py ## to seprate HC and PD for GM and WM
    5. Run ## $python3 Train_test,py  ## to split data into train and test data. 
            (Change the desire path inside Train_test file)
            
### CNN model
    1. 2D CNN
        
    
