import unittest
import os
import os.path as osp
from dl.main import main
from dl import args


class MyTestCase(unittest.TestCase):
    def test_main(self):
        main_script_dir = "../dl/"
        os.chdir(osp.dirname(main_script_dir))

        args.max_epochs = 2
        main(args)

    def test_eval(self):
        main_script_dir = "../dl/"
        os.chdir(osp.dirname(main_script_dir))

        args.num_gpus = 0
        args.model_num = 1
        args.mode = "eval"

        main(args)


if __name__ == '__main__':
    unittest.main()
