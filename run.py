import os
import glob
import argparse

from .app import GroupAlbum

def main(args):
    assert os.path.exists(args.input), f'Directory not exists: {args.input}'
    image_paths = []
    for img_ext in args.img_exts:
        image_paths += glob.glob(os.path.join(args.input, '**/*.' + img_ext), recursive=True)
    nrof_image = len(image_paths)
    if not nrof_image:
        print(f'Found no image files with extension {args.img_exts} in {args.input}')
        return
    worker = GroupAlbum(args.output, args.debug)
    worker.run(image_paths)
    worker.save()

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, 
        help='Input directory that has image files.')
    parser.add_argument('-o', '--output', type=str, default='./output', 
        help='Output directory to save image files')
    parser.add_argument('-d', '--debug', action='store_true',
        help='Whether to debug.')
    parser.add_argument('--img-exts', default=['jpg'], nargs='+',
        help='Image file extensions.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



