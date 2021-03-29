import unittest
from dl import config
from dl.dataloader import Dataset, collate_fn, gen_label_weight
from torch.utils.data import DataLoader


class MyTestCase(unittest.TestCase):
    def test_dataloader(self):
        args = config.get_arguments()
        phase = "train"
        label_weight_dict = gen_label_weight(args)
        print(label_weight_dict)

        ds = Dataset(args, phase, device="cuda")
        if args.node_label_cols == "dynamic_cat":
            assert ds.n_label == 4, f"{ds.n_label}"
        elif args.node_label_cols == "extend_dynamic_cat":
            assert ds.n_label == 6, f"{ds.n_label}"
        print(f"Number of trees in train set: {len(ds)}")

        ds_loader = DataLoader(ds, args.batch_size, shuffle=True, collate_fn=collate_fn)
        print(f'Batch size: {args.batch_size}')

        i = 0
        for batch_id, batched_graph in enumerate(ds_loader):
            print(f"Batch: {batch_id}")
            print(batched_graph.number_of_nodes)
            print("Labels: ", batched_graph.ndata["label"])
            print("In degress: ", batched_graph.in_degrees())
            i += 1
        print(f'Number of batches: {i}')


if __name__ == '__main__':
    unittest.main()
