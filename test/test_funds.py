import unittest

from lilt_fine_tune.funsdloader import Funsd
import torch

class MyFunsdCase(unittest.TestCase):
    def test_funsd_loader(self):
        loader = Funsd()
        generator = loader._generate_examples("../data/models/FUNSD/testing_data/")
        self.assertListEqual(list(next(generator)[1].keys()), ['id', 'words', 'bboxes', 'ner_tags', 'image_path'])

        # for data in generator:
        #     print(data)


    def test_cuda(self):
        print(torch.cuda.memory_summary())

if __name__ == '__main__':
    unittest.main()
