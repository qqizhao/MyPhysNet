class PreprocessArgs():
    def __init__(self):
        
        self.path = {
            'processed_data_path': '/home/robo/zhao_code/rPPG/MyPhysNet/processed_data',
            'file_list_name': 'file_list.csv',
        }
    
        self.config_preprocess = {
            'Crop_Face': {
                'Do_Crop_Face': True,
                'Backend': 'HC',
                'Use_Large_Face_Box': False,
                'Large_Box_Coef': 1.0,
                'Detection': {
                    'Do_Dynamic_Detection': False,  
                    'Dynamic_Detection_Frequency': 1,
                    'Use_Median_Face_Box': False,
                },
            },
            'Resize': {
                'W': 128,
                'H': 128,
            },
            'Data_Type': ['DiffNormalized',],   # if use physnet, should be ['DiffNormalized']
            'Label_Type': 'DiffNormalized',     
            'Chunk_Length': 128,
            'Do_Chunk': True,
        }

class UBFCArgs(PreprocessArgs):
    def __init__(self):
        super(UBFCArgs, self).__init__()
        
        self.dataset_name = 'UBFC-rPPG'
        self.path['raw_dataset_path'] = '/home/robo/zhao_code/rPPG/MyPhysNet/data/UBFC-rPPG'
        self.split_ratio = 0.5
        
        # self.do_preprocess = False
        self.do_preprocess = True      # if True, preprocess the data
        
        
class PureArgs(PreprocessArgs):
    def __init__(self):
        super(PureArgs, self).__init__()
        
        self.dataset_name = 'PURE'
        self.path['raw_dataset_path'] = '/home/robo/zhao_code/rPPG/MyPhysNet/data/PURE'
        self.split_ratio = 1
        
        # self.do_preprocess = False
        self.do_preprocess = True      # if True, preprocess the data
 
 
class TrainArgs():
    def __init__(self):
        
        self.mode = 'train_test'
        # self.mode = 'only_test'
        
        self.epochs =1
        self.lr = 1e-3
        self.batch_size = 2
        
        self.logs_path = './logs'
        self.checkpoints_path = './checkpoints'
        self.freq_save_model = 1
        
        # test config
        self.diff_flag = True         
        self.detrend_flag = True
        self.use_bandpass = True
        self.post_hr_method = 'FFT'   # 'Peak'
        self.Fs = 30
        
        self.preprocess_data_type = 'DiffNormalized'
        self.preprocess_label_type = 'DiffNormalized'