class PreprocessArgs():
    def __init__(self):
        
        self.path = {
            'raw_dataset_path': '/home/robo/zhao_code/rPPG/MyPhysNet/dataset',
            'processed_data_path': '/home/robo/zhao_code/rPPG/MyPhysNet/processed_data/UBFC-rPPG',
            'file_list_path': '/home/robo/zhao_code/rPPG/MyPhysNet/processed_data/DataFileList/file_list.csv',
        }
        
        self.dataset_name = 'UBFC-rPPG'
        # self.do_preprocess = False
        self.do_preprocess = True      # if True, preprocess the data
        
        self.split_ratio = 1
        
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
        
        
class TrainArgs():
    def __init__(self):
        
        # self.mode = 'train_test'
        self.mode = 'only_test'
        
        self.epochs =1
        self.lr = 1e-3
        self.batch_size = 6
        
        self.logs_path = './logs'
        self.checkpoints_path = './checkpoints'
        self.freq_save_model = 1
        
        # test config
        self.post = 'fft'
        self.diff = True
        self.detrend = True
        self.Fs = 30
        self.trans = None
        
        self.preprocess_data_type = 'Raw'
        self.preprocess_label_type = 'Raw'