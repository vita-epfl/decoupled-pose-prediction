

class args():
    def __init__(self):
        self.jaad_dataset = '' #folder containing parsed jaad annotations (used when first time loading data)
        self.dtype        = 'train'
        self.from_file    = False #read dataset from csv file or reprocess data
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
        self.batch_size     = 100 #70
        self.n_epochs       = 1000
        self.hidden_size    = 256  
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 14
        self.stride = 4
        self.skip   = 1
        self.task   = 'mask'
        self.use_scenes = False
        self.lr = 0.01
        #self.lr = 1.
        self.scalex = 1280.
        self.scaley = 720.
        self.scale =  False #True if true it scales the poses between 0 and 1 
   

