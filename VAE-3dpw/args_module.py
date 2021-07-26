class args():
    def __init__(self):
        self.jaad_dataset = '/data/smailait-data/JAAD/processed_annotations'
        self.dtype        = 'train'
        self.from_file    = False 
        self.save         = True
        self.file         = '/data/smailait-data/jaad_train_16_16.csv'
        self.save_path    = '/data/smailait-data/jaad_train_16_16.csv'
        self.model_path    = 'models/multitask_pv_lstm_trained.pkl'
        self.loader_workers = 1
        self.loader_shuffle = False #True 
        self.pin_memory     = False #True #False
        self.image_resize   = [240, 426]
        self.device         = 'cpu'
        self.batch_size     = 50
        self.n_epochs       = 3000
        self.hidden_size    = 1000
        self.hardtanh_limit = 100
        self.input  = 16
        self.output = 14
        self.stride = 16
        self.skip   = 1
        self.task   = 'pose'
        self.use_scenes = False
        self.lr = 0.001
        self.scale = False
        self.scalex = 1280
        self.scaley = 720 


