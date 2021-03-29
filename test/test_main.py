import unittest
import os
import os.path as osp
from dl.main import main
from dl import args


class MyTestCase(unittest.TestCase):
    def test_main(self):
        main_script_dir = "../dl/"
        os.chdir(osp.dirname(main_script_dir))
        print(os.getcwd())

        args.max_epochs = 2
        main(args)


if __name__ == '__main__':
    unittest.main()
