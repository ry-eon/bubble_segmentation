import argparse
import os, errno
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import random as rnd
import string
import sys

from tqdm import tqdm
from trdg.string_generator import create_strings_from_dict

from trdg.utils import load_dict, load_fonts
from trdg.data_generator import FakeTextDataGenerator
from multiprocessing import Pool
from trdg.stack import make_mask

def margins(margin):
    margins = margin.split(",")
    if len(margins) == 1:
        return [int(margins[0])] * 4
    return [int(m) for m in margins]


def parse_arguments():
    """
        Parse the command line arguments of the program.
    """

    parser = argparse.ArgumentParser(
        description="Generate synthetic text data for text recognition."
    )
    parser.add_argument(
        "--output_dir", type=str, nargs="?", help="The output directory", default="out/"
    )
    parser.add_argument(
        "-i",
        "--input_file",
        type=str,
        nargs="?",
        help="When set, this argument uses a specified text file as source for the text",
        default="",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        nargs="?",
        help="The language to use, should be fr (French), en (English), es (Spanish), de (German), cn (Chinese), or hi (Hindi)",
        default="ko",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        nargs="?",
        help="The number of images to be created.",
        required=True,
    )
    parser.add_argument(
        "-rs",
        "--random_sequences",
        action="store_true",
        help="Use random sequences as the source text for the generation. Set '-let','-num','-sym' to use letters/numbers/symbols. If none specified, using all three.",
        default=False,
    )
    parser.add_argument(
        "-let",
        "--include_letters",
        action="store_true",
        help="Define if random sequences should contain letters. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-num",
        "--include_numbers",
        action="store_true",
        help="Define if random sequences should contain numbers. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-sym",
        "--include_symbols",
        action="store_true",
        help="Define if random sequences should contain symbols. Only works with -rs",
        default=False,
    )
    parser.add_argument(
        "-w",
        "--length",
        type=int,
        nargs="?",
        help="Define how many words should be included in each generated sample. If the text source is Wikipedia, this is the MINIMUM length",
        default=1,
    )
    parser.add_argument(
        "-r",
        "--random",
        action="store_true",
        help="Define if the produced string will have variable word count (with --length being the maximum)",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--format",
        type=int,
        nargs="?",
        help="Define the height of the produced images if horizontal, else the width",
        default=30,
    )
    parser.add_argument(
        "-t",
        "--thread_count",
        type=int,
        nargs="?",
        help="Define the number of thread to use for image generation",
        default=1,
    )
    parser.add_argument(
        "-e",
        "--extension",
        type=str,
        nargs="?",
        help="Define the extension to save the image with",
        default="png",
    )
    parser.add_argument(
        "-k",
        "--skew_angle",
        type=int,
        nargs="?",
        help="Define skewing angle of the generated text. In positive degrees",
        default=0,
    )
    parser.add_argument(
        "-rk",
        "--random_skew",
        action="store_true",
        help="When set, the skew angle will be randomized between the value set with -k and it's opposite",
        default=False,
    )
    parser.add_argument(
        "-bl",
        "--blur",
        type=int,
        nargs="?",
        help="Apply gaussian blur to the resulting sample. Should be an integer defining the blur radius",
        default=0,
    )
    parser.add_argument(
        "-rbl",
        "--random_blur",
        action="store_true",
        help="When set, the blur radius will be randomized between 0 and -bl.",
        default=False,
    )
    parser.add_argument(
        "-b",
        "--background",
        type=int,
        nargs="?",
        help="Define what kind of background to use. 0: Gaussian Noise, 1: Plain white, 2: Quasicrystal, 3: Image",
        default=1,
    )
    parser.add_argument(
        "-na",
        "--name_format",
        type=int,
        help="Define how the produced files will be named. 0: [TEXT]_[ID].[EXT], 1: [ID]_[TEXT].[EXT] 2: [ID].[EXT] + one file labels.txt containing id-to-label mappings",
        default=0,
    )
    parser.add_argument(
        "-om",
        "--output_mask",
        type=int,
        help="Define if the generator will return masks for the text",
        default=0,
    )
    parser.add_argument(
        "-d",
        "--distorsion",
        type=int,
        nargs="?",
        help="Define a distorsion applied to the resulting image. 0: None (Default), 1: Sine wave, 2: Cosine wave, 3: Random",
        default=0,
    )
    parser.add_argument(
        "-do",
        "--distorsion_orientation",
        type=int,
        nargs="?",
        help="Define the distorsion's orientation. Only used if -d is specified. 0: Vertical (Up and down), 1: Horizontal (Left and Right), 2: Both",
        default=0,
    )
    parser.add_argument(
        "-wd",
        "--width",
        type=int,
        nargs="?",
        help="Define the width of the resulting image. If not set it will be the width of the text + 10. If the width of the generated text is bigger that number will be used",
        default=-1,
    )
    parser.add_argument(
        "-al",
        "--alignment",
        type=int,
        nargs="?",
        help="Define the alignment of the text in the image. Only used if the width parameter is set. 0: left, 1: center, 2: right",
        default=1,
    )
    parser.add_argument(
        "-or",
        "--orientation",
        type=int,
        nargs="?",
        help="Define the orientation of the text. 0: Horizontal, 1: Vertical",
        default=0,
    )
    parser.add_argument(
        "-tc",
        "--text_color",
        type=str,
        nargs="?",
        help="Define the text's color, should be either a single hex color or a range in the ?,? format.",
        default="#282828",
    )
    parser.add_argument(
        "-sw",
        "--space_width",
        type=float,
        nargs="?",
        help="Define the width of the spaces between words. 2.0 means twice the normal space width",
        default=1.0,
    )
    parser.add_argument(
        "-cs",
        "--character_spacing",
        type=int,
        nargs="?",
        help="Define the width of the spaces between characters. 2 means two pixels",
        default=0,
    )
    parser.add_argument(
        "-m",
        "--margins",
        type=margins,
        nargs="?",
        help="Define the margins around the text when rendered. In pixels",
        default=(5, 5, 5, 5),
    )
    parser.add_argument(
        "-fi",
        "--fit",
        action="store_true",
        help="Apply a tight crop around the rendered text",
        default=False,
    )
    parser.add_argument(
        "-id",
        "--image_dir",
        type=str,
        nargs="?",
        help="Define an image directory to use when background is set to image",
        default=os.path.join(os.path.split(os.path.realpath(__file__))[0], "images"),
    )
    parser.add_argument(
        "-ca",
        "--case",
        type=str,
        nargs="?",
        help="Generate upper or lowercase only. arguments: upper or lower. Example: --case upper",
    )
    parser.add_argument(
        "-ws",
        "--word_split",
        action="store_true",
        help="Split on words instead of on characters (preserves ligatures, no character spacing)",
        default=False,
    )
    parser.add_argument(
        "-stw",
        "--stroke_width",
        type=int, 
        nargs="?",
        help="Define the width of the strokes",
        default=0,
    )
    parser.add_argument(
        "-stf",
        "--stroke_fill",
        type=str, 
        nargs="?",
        help="Define the color of the contour of the strokes, if stroke_width is bigger than 0",
        default="#282828",
    )
    parser.add_argument(
        "-im",
        "--image_mode",
        type=str,
        nargs="?",
        help="Define the image mode to be used. RGB is default, L means 8-bit grayscale images, 1 means 1-bit binary images stored with one pixel per byte, etc.",
        default="RGB",
    )
    parser.add_argument(
        "--save_dir", type=str, nargs="?", help="The save directory", default="out/"
    )
    parser.add_argument(
        "-sn",
        type=str,
        nargs="?",
        help="Define how the produced mask will be named",
        default='mask',
    )
    parser.add_argument(
        "-mt",
        "--mask_term",
        type=int,
        nargs="?",
        help="Defines how many images are used in a row",
        default=2,
    )
    parser.add_argument(
        "-mw",
        "--mask_width",
        type=int,
        nargs="?",
        help="Define the width of the mask image.",
        default=3118,
    )
    parser.add_argument(
        "-mh",
        "--mask_height",
        type=int,
        nargs="?",
        help="Define the height of the mask image. ",
        default=4582,
    )

    return parser.parse_args()


def main():
    """
        Description: Main function
    """

    # Argument parsing
    args = parse_arguments()

    # Create the directory if it does not exist.
    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    # Creating word list

    print("dict2")
    lang_dict = load_dict(args.language)

    # Create font (path) list

    print("load ko fonts")
    fonts = load_fonts(args.language)

    # Creating synthetic sentences (or word)
    strings = []


    strings = create_strings_from_dict(
        args.length, args.random, args.count, lang_dict
    )

    if args.case == "upper":
        strings = [x.upper() for x in strings]
    if args.case == "lower":
        strings = [x.lower() for x in strings]

    string_count = len(strings)
    try :
        fonts = [x for x in fonts if x != '../trdg/fonts/ko/.DS_Store']
        p = Pool(args.thread_count)
        font_temp = [fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)]

        #assert False
        for _ in tqdm(
                p.imap_unordered(
                    FakeTextDataGenerator.generate_from_tuple,
                    zip(
                        [i for i in range(0, string_count)],
                        strings,
                        font_temp,#[fonts[rnd.randrange(0, len(fonts))] for _ in range(0, string_count)],
                        [args.output_dir] * string_count,
                        [args.format] * string_count,
                        [args.extension] * string_count,
                        [args.skew_angle] * string_count,
                        [args.random_skew] * string_count,
                        [args.blur] * string_count,
                        [args.random_blur] * string_count,
                        [args.background] * string_count,
                        [args.distorsion] * string_count,
                        [args.distorsion_orientation] * string_count,
                        [args.name_format] * string_count,
                        [args.width] * string_count,
                        [args.alignment] * string_count,
                        [args.text_color] * string_count,
                        [args.orientation] * string_count,
                        [args.space_width] * string_count,
                        [args.character_spacing] * string_count,
                        [args.margins] * string_count,
                        [args.fit] * string_count,
                        [args.output_mask] * string_count,
                        [args.word_split] * string_count,
                        [args.image_dir] * string_count,
                        [args.stroke_width] * string_count,
                        [args.stroke_fill] * string_count,
                        [args.image_mode] * string_count,
                    ),
                ),
                total=args.count,
        ):
            pass
        p.terminate()
    except:
        print("parameter error!")
        #print(font_temp)

    if args.name_format == 2:
        # Create file with filename-to-label connections
        with open(
            os.path.join(args.output_dir, "labels.txt"), "w", encoding="utf8"
        ) as f:
            for i in range(string_count):
                file_name = str(i) + "." + args.extension
                if args.space_width == 0:
                    file_name = file_name.replace(" ", "")
                f.write("{} {}\n".format(file_name, strings[i]))

    make_mask(args.output_dir,args.save_dir,args.mask_width,args.mask_height,args.sn)
if __name__ == "__main__":
    main()
