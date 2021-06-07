

class args():
    def __init__(self):
        self.jaad_dataset = '/home/parsaeif/bounding-box-prediction-bodyposes/JAAD/processed_annotations' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'train'
        self.from_file    = True #read dataset from csv file or reprocess data
        self.save         = False
        self.file         = 'jaad_train_16_16.csv'
        self.save_path    = 'jaad_train_16_16.csv'
        self.model_path    = 'models/multitask_pv_lstm_trained.pkl'
        self.loader_workers = 10
        self.loader_shuffle = True
        self.pin_memory     = False
        self.image_resize   = [240, 426]
        self.device         = 'cuda'
        #self.device         = 'cpu'
        self.batch_size     = 600 #70
        self.n_epochs       = 100000
        self.hidden_size    = 128  
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 14
        self.stride = 4
        self.skip   = 1
        self.task   = 'mask'
        self.use_scenes = False
        self.lr = 0.001
        #self.lr = 1.

