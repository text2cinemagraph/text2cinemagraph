import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None

    if 'motion' in opt.dataset_name:
        from data.custom_dataset import MotionPredictionDataset
        dataset = MotionPredictionDataset()
    elif 'frame' in opt.dataset_name:
        from data.custom_dataset import FramePredictionDataset
        dataset = FramePredictionDataset()
    
    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        worker_init_fn = self.dataset.worker_init_fn if hasattr(self.dataset, 'worker_init_fn') else None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=True,
            worker_init_fn=worker_init_fn)

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


