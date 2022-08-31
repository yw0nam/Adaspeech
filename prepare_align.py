import argparse

import yaml

from preprocessor import ljspeech, aishell3, libritts, kss, visual_novel, kokoro, visual_novel_non_parse


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "KSS" in config["dataset"]:
        kss.prepare_align(config)
    elif "visual_novel" in config['dataset']:
        visual_novel.prepare_align(config)
    if "kokoro" in config['dataset']:
        kokoro.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
