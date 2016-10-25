from paddle.utils.preprocess_img import ImageClassificationDatasetCreater
from optparse import OptionParser


def option_parser():
    parser = OptionParser(usage="usage: python preprcoess.py "\
                          "-i data_dir [options]")
    parser.add_option("-i", "--input", action="store",
                      dest="input", help="Input data directory.")
    parser.add_option("-s", "--size", action="store",
                      dest="size", help="Processed image size.")
    parser.add_option("-c", "--color", action="store",
                      dest="color", help="whether to use color images.")
    return parser.parse_args()

if __name__ == '__main__':
     options, args = option_parser()
     data_dir = options.input
     processed_image_size = int(options.size)
     color = options.color == "1"
     data_creator = ImageClassificationDatasetCreater(data_dir,
                                                      processed_image_size,
                                                      color)
     data_creator.num_per_batch = 1000
     data_creator.overwrite = True
     data_creator.create_batches()
