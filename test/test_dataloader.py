import unittest
from models import config
from models.dataloader import Dataset, collate_fn
from torch.utils.data import DataLoader


class MyTestCase(unittest.TestCase):
    def test_dataloader(self):
        args = config.get_arguments()
        phase = "train"

        ds = Dataset(args, phase=phase)
        print(f"Number of trees in train set: {len(ds)}")

        ds_loader = DataLoader(ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
        print(f'Batch size: {args.batch_size}')

        i = 0
        for batch_id, batched_graph in enumerate(ds_loader):
            print(f"Batch: {batch_id}")
            print(batched_graph.number_of_nodes)
        print(f'Number of batches: {i}')


if __name__ == '__main__':
    unittest.main()
